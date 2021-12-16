package org.sgd

import kotlinx.atomicfu.atomic
import java.lang.invoke.MethodHandles
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min
import kotlin.math.pow
import kotlin.random.Random


class SGDResult(val w: TypeArray, val timeNsToWeights: LinkedHashMap<Long, TypeArray>)

interface SGDSolver {
    fun solve(loss: Model, testLoss: Model, targetLoss: Type): SGDResult
}


inline fun measureIterations(
    testLoss: Model,
    targetLoss: Type,
    crossinline accessWeights: () -> TypeArray,
    iterations: (AtomicBoolean) -> Unit
): LinkedHashMap<Long, TypeArray> {
    val stop = AtomicBoolean(false)
    val timeNsToWeights = linkedMapOf<Long, TypeArray>()
//    val startNs = System.nanoTime()
    val scheduledThreadPool = Executors.newScheduledThreadPool(1)
    val future = scheduledThreadPool.scheduleAtFixedRate({
        val weights = accessWeights()
//        val timeNs = System.nanoTime() - startNs
//        timeNsToWeights[timeNs] = weights
        val currentLoss = testLoss.loss(weights)
        if (currentLoss <= targetLoss) {
            stop.set(true)
        }
    }, 0, 25, TimeUnit.MILLISECONDS)
    iterations(stop)
    future.cancel(false)
    scheduledThreadPool.shutdownNow()
    return timeNsToWeights
}

class SequentialSGDSolver(
    private val iterations: Int,
    private val alpha: Type,
    private val stepDecay: Type
) : SGDSolver {
    override fun solve(loss: Model, testLoss: Model, targetLoss: Type): SGDResult {
        val w = loss.createWeights()
        val fs = loss.points
        val random = Random
        val n = fs.size
        val timeToLoss = measureIterations(testLoss, targetLoss, { w.copyOf() }) {
            var learningRate = alpha
            repeat(iterations) {
                repeat(n) {
                    val i = random.nextInt(n)
                    loss.gradientStep(fs[i], w, learningRate)
                }
                learningRate *= stepDecay
            }
        }
        return SGDResult(w, timeToLoss)
    }
}

class ParallelSGDSolver(
    private val alpha: Type,
    private val threads: Int,
    private val stepDecay: Type
) : SGDSolver {
    private lateinit var stop: AtomicBoolean

    override fun solve(loss: Model, testLoss: Model, targetLoss: Type): SGDResult {
        val w = loss.createWeights()
        val tasks = numaConfig.values.flatten().take(threads).map { threadId -> Thread { threadSolve(w, threadId, loss) } }
        val timeToLoss = measureIterations(testLoss, targetLoss, { w.copyOf() }) {
            stop = it
            tasks
                .onEach { it.start() }
                .forEach { it.join() }
        }

        return SGDResult(w, timeToLoss)
    }

    private fun threadSolve(w: TypeArray, threadId: Int, loss: Model) {
        bindCurrentThreadToCpu(threadId)
        val points = loss.points
        var learningRate = alpha
        val stop = stop
        val n = points.size
        while (true) {
            repeat(n) {
                if (stop.get()) return
                val i = Random.nextInt(n)
                loss.gradientStep(points[i], w, learningRate)
            }
            learningRate *= stepDecay
        }
    }
}


private class ClusterData(val w: TypeArray, val wPrev: TypeArray, val clusterId: Int) {
    @Volatile
    var nextThread: Int = 0
}

fun extractClusters(threads: Int, threadsPerCluster: Int): MutableList<List<Int>> {
    val clusters = mutableListOf<List<Int>>()
    var cores = threads
    for (cpus in numaConfig.values) {
        var i = 0
        while (cores > 0 && i + min(threadsPerCluster, cores) <= cpus.size) {
            val portion = min(threadsPerCluster, cores)
            cores -= portion
            clusters.add(cpus.subList(i, i + portion).toList())
            i += portion
        }
    }
    return clusters
}

class ClusterParallelSGDSolver(
    private val alpha: Type,
    private val threads: Int,
    private val stepDecay: Type,
    threadsPerCluster: Int,
    private val stepsBeforeTokenPass: Int = 10000
) : SGDSolver {
    private val clusters: List<List<Int>>
    private lateinit var stop: AtomicBoolean

    init {
        check(threads <= numaConfig.values.sumOf { it.size })
        val clusters = extractClusters(threads, threadsPerCluster)
        this.clusters = clusters
    }

    private val AA = MethodHandles.arrayElementVarHandle(TypeArray::class.java)
    private val beta = findRoot { x -> x.pow(clusters.size) + x - 1.0 }.toType()
    private val lambda = 1 - beta.pow(clusters.size - 1)
    private val token = atomic(0)
    private lateinit var test: Model

    private lateinit var clustersData: List<ClusterData>

    override fun solve(loss: Model, testLoss: Model, targetLoss: Type): SGDResult {
        val w = loss.createWeights()
        clustersData = clusters.indices.map { i -> ClusterData(w.copyOf(), w.copyOf(), i) }
        test = testLoss

        val tasks = clusters.withIndex().flatMap { (clusterId, cores) ->
            cores.indices
                .map { i -> Thread { threadSolve(clustersData[clusterId], cores[i], cores[(i + 1) % cores.size], loss) } }
        }
        val timeToLoss = measureIterations(testLoss, targetLoss, { getResults(TypeArray(w.size)) }) {
            stop = it
            tasks
                .onEach { it.start() }
                .forEach { it.join() }
        }

        return SGDResult(getResults(w), timeToLoss)
    }

    private fun getResults(buffer: TypeArray): TypeArray {
        if (clustersData.size == 1) {
            return clustersData[0].w.copyInto(buffer)
        }
        buffer.resetToZero()
        for (data in clustersData) {
            buffer.add(data.w)
        }
        buffer.divide(clusters.size.toType())
        return buffer
    }

    private fun threadSolve(clusterData: ClusterData, threadId: Int, nextThreadId: Int, loss: Model) {
        bindCurrentThreadToCpu(threadId)
        val w = clusterData.w
        val points = loss.points
        var learningRate = alpha

        var step = 0
        var locked = false
        val shouldSync = clusters.size > 1
        val stop = stop
        val n = points.size
        val stepsBeforeTokenPass = stepsBeforeTokenPass
        val threads = threads
        val block = n / threads
        val start = block * threadId
        val end = if (threadId == threads - 1) n else start + block
        while (true) {
            repeat(end - start) {
                if (stop.get()) {
                    if (locked) {
                        releaseToken(clusterData, nextThreadId)
                    }
                    return
                }
                loss.gradientStep(points[start + it], w, learningRate)

                if (shouldSync) {
                    if (!locked) {
                        val tokenValue = token.value
                        if (tokenValue >= 0 && clusterData.clusterId == tokenValue && threadId == clusterData.nextThread) {
                            assert(token.compareAndSet(tokenValue, -(clusterData.clusterId + 1)))
                            locked = true
                            step = 0
                            syncWithNext(clusterData)
                        }
                    }

                    step++

                    if (locked && step == stepsBeforeTokenPass) {
                        releaseToken(clusterData, nextThreadId)
                        locked = false
                    }
                }
            }
            learningRate *= stepDecay
        }
    }

    private fun syncWithNext(clusterData: ClusterData) {
        if (clusters.size == 1) return
        val next = (clusterData.clusterId + 1) % clusters.size
        val w = clusterData.w
        val wPrev = clusterData.wPrev
        repeat(w.size) { i ->
            val delta = w[i] - wPrev[i]
            val nextValue = AA.getAndAdd(clustersData[next].w, i, beta * delta) as Type
            wPrev[i] = lambda * nextValue + (1 - lambda) * wPrev[i] + beta * delta
        }
    }

    private fun releaseToken(clusterData: ClusterData, nextThreadId: Int) {
        clusterData.nextThread = nextThreadId
        assert(token.compareAndSet(-(clusterData.clusterId + 1), (clusterData.clusterId + 1) % clusters.size))
    }
}

/**
 * @param f a monotonically non-decreasing function
 */
private inline fun findRoot(f: (Double) -> Double): Double {
    var l = 0.0
    var r = 1.0
    val eps = 1e-10
    while (r - l > eps) {
        val m = (r + l) / 2
        if (f(m) < 0.0) {
            l = m
        } else {
            r = m
        }
    }
    return (l + r) / 2
}
