package org.sgd

import kotlinx.atomicfu.atomic
import java.lang.invoke.MethodHandles
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.pow


class SGDResult(val w: Weights, val timeNsToWeights: LinkedHashMap<Long, Weights>)

interface SGDSolver {
    fun solve(loss: DataSetLoss, initial: Weights): SGDResult
}

inline fun scheduledTask(
    timeNsToWeights: LinkedHashMap<Long, Weights>,
    crossinline accessWeights: () -> Weights
): () -> Boolean {
    val startNs = System.nanoTime()
    val scheduler = Executors.newScheduledThreadPool(1)
    val future = scheduler.scheduleAtFixedRate({
        timeNsToWeights[System.nanoTime() - startNs] = accessWeights()
    }, 0, 1, TimeUnit.SECONDS)
    return { future.cancel(true) }
}

class SequentialSGDSolver(
    private val iterations: Int,
    private val alpha: Double,
    private val stepDecay: Double
) : SGDSolver {
    override fun solve(loss: DataSetLoss, initial: Weights): SGDResult {
        val w = initial
        val buffer = DoubleArray(w.size)
        val fs = loss.pointLoss.toMutableList()
        val timeToLoss = linkedMapOf<Long, Weights>()

        val stop = scheduledTask(timeToLoss) { w.copyOf() }
        var learningRate = alpha
        repeat(iterations) {
            fs.shuffle()
            repeat(fs.size) {
                w.subtract(fs[it].grad(w, buffer).multiply(learningRate))
            }
            learningRate *= stepDecay
        }
        stop()
        return SGDResult(w, timeToLoss)
    }
}

class ParallelSGDSolver(
    private val iterations: Int,
    private val alpha: Double,
    private val threads: Int,
    private val stepDecay: Double
) : SGDSolver {
    override fun solve(loss: DataSetLoss, initial: Weights): SGDResult {
        val timeToLoss = linkedMapOf<Long, Weights>()

        val tasks = List(threads) { Thread { threadSolve(initial, loss) } }
        val stop = scheduledTask(timeToLoss) { initial.copyOf() }
        tasks
            .onEach { it.start() }
            .forEach { it.join() }
        stop()

        return SGDResult(initial, timeToLoss)
    }

    private fun threadSolve(w: Weights, loss: DataSetLoss) {
        val buffer = Weights(w.size)
        val points = loss.pointLoss.toMutableList()
        var learningRate = alpha
        repeat(iterations) {
            points.shuffle()
            repeat(points.size) {
                points[it].grad(w, buffer).multiply(learningRate)
                w.subtract(buffer)
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
    private val iterations: Int,
    private val alpha: Double,
    private val threads: Int,
    private val stepDecay: Double,
    private val stepsBeforeTokenPass: Int = 1000
) : SGDSolver {
    private val clusters: List<List<Int>>

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


    override fun solve(loss: DataSetLoss, initial: Weights): SGDResult {
        clustersData = clusters.indices.map { i -> ClusterData(initial.copyOf(), initial.copyOf(), i) }

        val timeToLoss = linkedMapOf<Long, Weights>()

        var activeCores = 0
        val tasks = clusters.withIndex().flatMap { (clusterId, cores) ->
            cores.indices
                .take(threads - activeCores)
                .map { i ->
                    Thread {
                        threadSolve(
                            clustersData[clusterId],
                            cores[i],
                            cores[(i + 1) % cores.size],
                            loss
                        )
                    }
                }
                .also { activeCores += it.size }
        }
        val stop = scheduledTask(timeToLoss) { getResults(DoubleArray(initial.size)) }
        tasks
            .onEach { it.start() }
            .forEach { it.join() }
        stop()

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
        val buffer = Weights(w.size)
        val points = loss.pointLoss.toMutableList()
        var learningRate = alpha

        var step = 0
        var locked = false
        val shouldSync = clusters.size > 1
        repeat(iterations) {
            points.shuffle()

            repeat(points.size) {
                points[it].grad(w, buffer).multiply(learningRate)
                w.subtract(buffer)

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
        repeat(clusterData.w.size) { i ->
            val delta = clusterData.w[i] - clusterData.wPrev[i]
            val nextValue = AA.getAndAdd(clustersData[next].w, i, beta * delta) as Double
            clusterData.wPrev[i] = lambda * nextValue + (1 - lambda) * clusterData.wPrev[i] + beta * delta
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
