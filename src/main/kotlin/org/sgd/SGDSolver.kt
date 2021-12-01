package org.sgd

import kotlinx.atomicfu.atomic
import java.lang.invoke.MethodHandles
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.pow


class SGDResult(val w: Weights, val timeNsToWeights: LinkedHashMap<Long, Weights>)

interface SGDSolver {
    fun solve(loss: DataSetLoss, testLoss: DataSetLoss, initial: Weights, targetLoss: LossValue): SGDResult
}


inline fun measureIterations(
    testLoss: DataSetLoss,
    targetLoss: LossValue,
    crossinline accessWeights: () -> Weights,
    iterations: (AtomicBoolean) -> Unit
): LinkedHashMap<Long, Weights> {
    val stop = AtomicBoolean(false)
    val timeNsToWeights = linkedMapOf<Long, Weights>()

    val startNs = System.nanoTime()

    val future = Executors.newScheduledThreadPool(1).scheduleAtFixedRate({
        val weights = accessWeights()
        val timeNs = System.nanoTime() - startNs
        timeNsToWeights[timeNs] = weights
        val currentLoss = testLoss.loss(weights)
        if (currentLoss <= targetLoss) {
            stop.set(true)
        }
    }, 0, 100, TimeUnit.MILLISECONDS)
    iterations(stop)
    future.cancel(false)
    return timeNsToWeights
}

class SequentialSGDSolver(
    private val iterations: Int,
    private val alpha: Double,
    private val stepDecay: Double
) : SGDSolver {
    override fun solve(loss: DataSetLoss, testLoss: DataSetLoss, initial: Weights, targetLoss: LossValue): SGDResult {
        val w = initial
        val fs = loss.pointLoss.toMutableList()
        val timeToLoss = measureIterations(testLoss, targetLoss, { w.copyOf() }) {
            var learningRate = alpha
            repeat(iterations) {
                fs.shuffle()
                repeat(fs.size) {
                    fs[it].gradientStep(w, learningRate)
                }
                learningRate *= stepDecay
            }
        }
        return SGDResult(w, timeToLoss)
    }
}

class ParallelSGDSolver(
    private val alpha: Double,
    private val threads: Int,
    private val stepDecay: Double
) : SGDSolver {
    private lateinit var stop: AtomicBoolean

    override fun solve(loss: DataSetLoss, testLoss: DataSetLoss, initial: Weights, targetLoss: LossValue): SGDResult {
        val tasks = numaConfig.values.flatten().take(threads).map { threadId -> Thread { threadSolve(initial, threadId, loss) } }
        val timeToLoss = measureIterations(testLoss, targetLoss, { initial.copyOf() }) {
            stop = it
            tasks
                .onEach { it.start() }
                .forEach { it.join() }
        }

        return SGDResult(initial, timeToLoss)
    }

    private fun threadSolve(w: Weights, threadId: Int, loss: DataSetLoss) {
        bindCurrentThreadToCpu(threadId)
        val points = loss.pointLoss.toMutableList()
        var learningRate = alpha
        val stop = stop
        while (!stop.get()) {
            points.shuffle()
            repeat(points.size) {
                points[it].gradientStep(w, learningRate)
            }
            learningRate *= stepDecay
        }
    }
}


private class ClusterData(val w: Weights, val wPrev: Weights, val clusterId: Int) {
    @Volatile
    var nextThread: Int = 0
}

class ClusterParallelSGDSolver(
    private val alpha: Double,
    private val threads: Int,
    private val stepDecay: Double,
    private val stepsBeforeTokenPass: Int = 1000
) : SGDSolver {
    private val clusters: List<List<Int>>
    private lateinit var stop: AtomicBoolean

    init {
        check(threads <= numaConfig.values.sumOf { it.size })
        var cores = 0
        clusters = numaConfig.filter { numaNode ->
            (cores < threads).also {
                cores += numaNode.value.size
            }
        }.values.toList()
    }

    private val AA = MethodHandles.arrayElementVarHandle(DoubleArray::class.java)
    private val beta = findRoot { x -> x.pow(clusters.size) + x - 1.0 }
    private val lambda = 1 - beta.pow(clusters.size - 1)
    private val token = atomic(0)

    @Volatile
    private lateinit var clustersData: List<ClusterData>

    override fun solve(loss: DataSetLoss, testLoss: DataSetLoss, initial: Weights, targetLoss: LossValue): SGDResult {
        clustersData = clusters.indices.map { i -> ClusterData(initial.copyOf(), initial.copyOf(), i) }

        var activeCores = 0
        val tasks = clusters.withIndex().flatMap { (clusterId, cores) ->
            cores.indices
                .take(threads - activeCores)
                .map { i -> Thread { threadSolve(clustersData[clusterId], cores[i], cores[(i + 1) % cores.size], loss) } }
                .also { activeCores += it.size }
        }
        val timeToLoss = measureIterations(testLoss, targetLoss, { getResults(DoubleArray(initial.size)) }) {
            stop = it
            tasks
                .onEach { it.start() }
                .forEach { it.join() }
        }

        return SGDResult(getResults(initial), timeToLoss)
    }

    private fun getResults(buffer: Weights): Weights {
        if (clustersData.size == 1) {
            return clustersData[0].w.copyInto(buffer)
        }
        buffer.resetToZero()
        for (data in clustersData) {
            buffer.add(data.w)
        }
        buffer.divide(clusters.size.toDouble())
        return buffer
    }

    private fun threadSolve(clusterData: ClusterData, threadId: Int, nextThreadId: Int, loss: DataSetLoss) {
        bindCurrentThreadToCpu(threadId)
        val w = clusterData.w
        val points = loss.pointLoss.toMutableList()
        var learningRate = alpha

        var step = 0
        var locked = false
        val shouldSync = clusters.size > 1
        val stop = stop
        while (!stop.get()) {
            points.shuffle()

            repeat(points.size) {
                points[it].gradientStep(w, learningRate)

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
        if (locked) {
            releaseToken(clusterData, nextThreadId)
        }
    }

    private fun syncWithNext(clusterData: ClusterData) {
        if (clusters.size == 1) return
        val next = (clusterData.clusterId + 1) % clusters.size
        val w = clusterData.w
        val wPrev = clusterData.wPrev
        repeat(w.size) { i ->
            val delta = w[i] - wPrev[i]
            val nextValue = AA.getAndAdd(clustersData[next].w, i, beta * delta) as Double
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
