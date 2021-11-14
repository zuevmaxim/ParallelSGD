package org.sgd

import kotlinx.atomicfu.atomic
import java.lang.invoke.MethodHandles
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.ceil
import kotlin.math.pow


class SGDResult(val w: Weights, val timeNsToWeights: LinkedHashMap<Long, Weights>)

interface SGDSolver {
    fun solve(loss: DataSetLoss, initial: Weights): SGDResult
}

inline fun scheduledTask(timeNsToWeights: LinkedHashMap<Long, Weights>, crossinline accessWeights: () -> Weights): () -> Boolean {
    val startNs = System.nanoTime()
    val scheduler = Executors.newScheduledThreadPool(1)
    val future = scheduler.scheduleAtFixedRate({
        timeNsToWeights[System.nanoTime() - startNs] = accessWeights()
    }, 0, 1, TimeUnit.SECONDS)
    return { future.cancel(true) }
}

class SequentialSGDSolver(
    private val iterations: Int,
    private val alpha: Double
) : SGDSolver {
    override fun solve(loss: DataSetLoss, initial: Weights): SGDResult {
        val w = initial
        val buffer = DoubleArray(w.size)
        val fs = loss.pointLoss.toMutableList()
        val timeToLoss = linkedMapOf<Long, Weights>()

        val stop = scheduledTask(timeToLoss) { w.copyOf() }
        repeat(iterations) {
            fs.shuffle()
            repeat(fs.size) {
                w.subtract(fs[it].grad(w, buffer).multiply(alpha))
            }
        }
        stop()
        return SGDResult(w, timeToLoss)
    }
}

class ParallelSGDSolver(
    private val iterations: Int,
    private val alpha: Double,
    private val threads: Int
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
        repeat(iterations) {
            points.shuffle()
            repeat(points.size) {
                points[it].grad(w, buffer).multiply(alpha)
                w.subtract(buffer)
            }
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
    private val clusters: Int,
    private val stepsBeforeTokenPass: Int = 1000
) : SGDSolver {
    private val AA = MethodHandles.arrayElementVarHandle(DoubleArray::class.java)
    private val beta = findRoot { x -> x.pow(clusters) + x - 1.0 }
    private val lambda = 1 - beta.pow(clusters - 1)
    private val token = atomic(0)

    @Volatile
    private lateinit var clustersData: List<ClusterData>


    override fun solve(loss: DataSetLoss, initial: Weights): SGDResult {
        clustersData = List(clusters) { i -> ClusterData(initial.copyOf(), initial.copyOf(), i) }

        val timeToLoss = linkedMapOf<Long, Weights>()

        val threadsPerCluster = ceil(threads / clusters.toDouble()).toInt()
        val tasks = List(threads) { i ->
            val clusterId = i / threadsPerCluster
            val threadId = i % threadsPerCluster
            val nextThreadId = ((i + 1) % threads) % threadsPerCluster
            Thread { threadSolve(clustersData[clusterId], threadId, nextThreadId, loss) }
        }
        val stop = scheduledTask(timeToLoss) { getResults(DoubleArray(initial.size)) }
        tasks
            .onEach { it.start() }
            .forEach { it.join() }
        stop()

        return SGDResult(getResults(initial), timeToLoss)
    }

    private fun getResults(buffer: Weights): Weights {
        buffer.resetToZero()
        for (data in clustersData) {
            buffer.add(data.w)
        }
        buffer.divide(clusters.toDouble())
        return buffer
    }

    private fun threadSolve(clusterData: ClusterData, threadId: Int, nextThreadId: Int, loss: DataSetLoss) {
        val buffer = Weights(clusterData.w.size)
        val points = loss.pointLoss.toMutableList()
        var step = 0
        var locked = false

        repeat(iterations) {
            points.shuffle()

            repeat(loss.pointLoss.size) {
                points[it].grad(clusterData.w, buffer).multiply(alpha)
                clusterData.w.subtract(buffer)

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
        if (locked) {
            releaseToken(clusterData, nextThreadId)
        }
    }

    private fun syncWithNext(clusterData: ClusterData) {
        val next = (clusterData.clusterId + 1) % clusters
        repeat(clusterData.w.size) { i ->
            val delta = clusterData.w[i] - clusterData.wPrev[i]
            val nextValue = AA.getAndAdd(clustersData[next].w, i, beta * delta) as Double
            clusterData.wPrev[i] = lambda * nextValue + (1 - lambda) * clusterData.wPrev[i] + beta * delta
        }
    }

    private fun releaseToken(clusterData: ClusterData, nextThreadId: Int) {
        clusterData.nextThread = nextThreadId
        assert(token.compareAndSet(-(clusterData.clusterId + 1), (clusterData.clusterId + 1) % clusters))
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
