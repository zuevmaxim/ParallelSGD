package org.sgd

import kotlinx.atomicfu.atomic
import java.lang.invoke.MethodHandles
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.math.ceil
import kotlin.math.pow
import kotlin.system.measureNanoTime

class SGDResult(val w: Weights, val timeNsToLoss: LinkedHashMap<Long, Double>)

interface SGDSolver {
    fun solve(loss: DataSetLoss, initial: Weights): SGDResult
}

class SimpleSGDSolver(
    private val iterations: Int,
    private val alpha: Double
) : SGDSolver {
    override fun solve(loss: DataSetLoss, initial: Weights): SGDResult {
        val w = initial
        val buffer = DoubleArray(w.size)
        val fs = loss.pointLoss.toMutableList()
        val timeToLoss = linkedMapOf<Long, Double>()

        var time = 0L
        repeat(iterations) {
            fs.shuffle()

            val timeNs = measureNanoTime {
                repeat(fs.size) {
                    w.subtract(fs[it].grad(w, buffer).multiply(alpha))
                }
            }

            time += timeNs
            timeToLoss[time] = loss.loss(w)
        }
        return SGDResult(w, timeToLoss)
    }
}

class SimpleParallelSGDSolver(
    private val iterations: Int,
    private val alpha: Double,
    private val threads: Int
) : SGDSolver {
    override fun solve(loss: DataSetLoss, initial: Weights): SGDResult {
        val timeToLoss = linkedMapOf<Long, Double>()
        val pool = Executors.newFixedThreadPool(threads)
        val buffers = List(threads) { DoubleArray(initial.size) }

        var time = 0L
        repeat(iterations) {
            (loss.pointLoss as MutableList).shuffle()

            val timeNs = measureNanoTime {
                pool.invokeAll(List(threads) { i -> Callable { threadSolve(initial, loss, buffers[i]) } })
                    .forEach { it.get() }
            }

            time += timeNs
            timeToLoss[time] = loss.loss(initial)
        }

        return SGDResult(initial, timeToLoss)
    }

    private fun threadSolve(w: Weights, loss: DataSetLoss, buffer: Weights) {
        repeat(loss.pointLoss.size) {
            w.subtract(loss.pointLoss[it].grad(w, buffer).multiply(alpha))
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

        val timeToLoss = linkedMapOf<Long, Double>()
        val pool = Executors.newFixedThreadPool(threads)
        val buffers = List(threads) { DoubleArray(initial.size) }

        var time = 0L
        repeat(iterations) {
            (loss.pointLoss as MutableList).shuffle()

            val timeNs = measureNanoTime {
                val threadsPerCluster = ceil(threads / clusters.toDouble()).toInt()
                pool.invokeAll(List(threads) { i ->
                    val clusterId = i / threadsPerCluster
                    val threadId = i % threadsPerCluster
                    val nextThreadId = ((i + 1) % threads) % threadsPerCluster
                    Callable { threadSolve(clustersData[clusterId], threadId, nextThreadId, loss, buffers[i]) }
                })
                    .forEach { it.get() }
            }

            time += timeNs
            timeToLoss[time] = loss.loss(getResults(initial))
        }

        return SGDResult(getResults(initial), timeToLoss)
    }

    private fun getResults(initial: Weights): Weights {
        initial.resetToZero()
        for (data in clustersData) {
            initial.add(data.w)
        }
        initial.divide(clusters.toDouble())
        return initial
    }

    private fun threadSolve(clusterData: ClusterData, threadId: Int, nextThreadId: Int, loss: DataSetLoss, buffer: Weights) {
        var step = 0
        var locked = false
        repeat(loss.pointLoss.size) {
            clusterData.w.subtract(loss.pointLoss[it].grad(clusterData.w, buffer).multiply(alpha))

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
