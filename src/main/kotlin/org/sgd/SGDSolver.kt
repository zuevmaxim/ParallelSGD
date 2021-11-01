package org.sgd

import kotlinx.atomicfu.atomic
import java.lang.invoke.MethodHandles
import kotlin.math.pow

class SGDResult(val w: Weights, val lossIterations: DoubleArray)

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
        val lossIterations = DoubleArray(iterations)
        repeat(iterations) { iteration ->
            fs.shuffle()
            repeat(fs.size) {
                w.subtract(fs[it].grad(w, buffer).multiply(alpha))
            }
            lossIterations[iteration] = loss.loss(w)
        }
        return SGDResult(w, lossIterations)
    }
}

class SimpleParallelSGDSolver(
    private val iterations: Int,
    private val alpha: Double,
    private val threads: Int
) : SGDSolver {
    override fun solve(loss: DataSetLoss, initial: Weights): SGDResult {
        val lossIterations = DoubleArray(iterations)
        val threads = List(threads) { Thread { threadSolve(initial, loss, lossIterations) } }
        threads.forEach { it.start() }
        threads.forEach { it.join() }
        return SGDResult(initial, lossIterations)
    }

    private fun threadSolve(w: Weights, loss: DataSetLoss, lossIterations: DoubleArray) {
        val buffer = DoubleArray(w.size)
        val fs = loss.pointLoss.toMutableList()
        repeat(iterations) { iteration ->
            fs.shuffle()
            repeat(fs.size) {
                fs[it].grad(w, buffer).multiply(alpha)
                repeat(buffer.size) { i ->
                    if (buffer[i] != 0.0) {
                        w[i] -= buffer[i]
                    }
                }
            }
            lossIterations[iteration] = loss.loss(w)
        }
    }
}

class ClusterParallelSGDSolver(
    private val iterations: Int,
    private val alpha: Double,
    private val threads: Int,
    private val clusters: Int,
    private val stepsBeforeTokenPass: Int = 1000
) : SGDSolver {
    private val AA = MethodHandles.arrayElementVarHandle(DoubleArray::class.java)
    private val beta = computeBeta()
    private val lambda = 1 - beta.pow(clusters - 1)

    private fun computeBeta(): Double {
        var l = 0.0
        var r = 1.0
        val eps = 1e-10
        while (r - l > eps) {
            val m = (r + l) / 2
            if (m.pow(clusters) + m < 1.0) {
                l = m
            } else {
                r = m
            }
        }
        return (l + r) / 2
    }

    private class ClusterData(val w: Weights, val wPrev: Weights, val clusterId: Int, @Volatile var nextThread: Int)

    private val token = atomic(-1)

    @Volatile
    private lateinit var clustersData: List<ClusterData>


    override fun solve(loss: DataSetLoss, initial: Weights): SGDResult {
        clustersData = List(clusters) { i -> ClusterData(initial.copyOf(), initial.copyOf(), i, 0) }

        val lossIterations = DoubleArray(iterations)
        val threadsPerCluster = threads / clusters
        val threads = List(threads) { i -> Thread { threadSolve(clustersData[i / threadsPerCluster], i % threadsPerCluster, (i + 1) % threadsPerCluster, loss, lossIterations) } }
        threads.forEach { it.start() }
        token.incrementAndGet()
        threads.forEach { it.join() }

        initial.resetToZero()
        for (data in clustersData) {
            initial.add(data.w)
        }
        initial.divide(clusters.toDouble())

        return SGDResult(initial, lossIterations)
    }

    private fun threadSolve(clusterData: ClusterData, threadId: Int, nextThreadId: Int, loss: DataSetLoss, lossIterations: DoubleArray) {
        val buffer = DoubleArray(clusterData.w.size)
        val fs = loss.pointLoss.toMutableList()
        var step = 0
        var locked = false
        repeat(iterations) { iteration ->
            fs.shuffle()
            repeat(fs.size) {
                fs[it].grad(clusterData.w, buffer).multiply(alpha)
                repeat(buffer.size) { i ->
                    if (buffer[i] != 0.0) {
                        clusterData.w[i] -= buffer[i]
                    }
                }

                if (!locked) {
                    var tokenValue = token.value
                    while (tokenValue >= 0 && clusterData.clusterId == tokenValue % clusters && threadId == clusterData.nextThread) {
                        if (token.compareAndSet(tokenValue, -(tokenValue + 1))) {
                            locked = true
                            break
                        }
                        tokenValue = token.value
                    }
                    if (locked) {
                        val next = (clusterData.clusterId + 1) % clusters
                        repeat(clusterData.w.size) { i ->
                            val delta = clusterData.w[i] - clusterData.wPrev[i]
                            val nextValue = AA.getAndAdd(clustersData[next].w, i, beta * delta) as Double
                            clusterData.wPrev[i] = lambda * nextValue + (1 - lambda) * clusterData.wPrev[i] + beta * delta
                        }
                        step = 0
                    }
                }

                step++

                if (locked && step == stepsBeforeTokenPass) {
                    clusterData.nextThread = nextThreadId
                    assert(token.compareAndSet(-(clusterData.clusterId + 1), (clusterData.clusterId + 1) % clusters))
                    locked = false
                }
            }
            lossIterations[iteration] = loss.loss(clusterData.w)
        }
    }
}
