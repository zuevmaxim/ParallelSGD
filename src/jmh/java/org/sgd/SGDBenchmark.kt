package org.sgd

import org.openjdk.jmh.annotations.*
import java.util.concurrent.TimeUnit

@State(Scope.Benchmark)
@Measurement(iterations = 5, time = 10, timeUnit = TimeUnit.SECONDS)
@BenchmarkMode(Mode.AverageTime)
@Fork(1)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
open class SGDBenchmark {
    var dataset: String = "rcv1"
    val method: String = "simple"
    val learningRate = params[dataset]!!["learningRate"] as Type
    val stepDecay = params[dataset]!!["stepDecay"] as Type
    val targetLoss = params[dataset]!!["targetLoss"] as Type

    @Param("1", "2", "4")
    var workingThreads: Int = 0

    private val loss = models[dataset]!!()


    @Benchmark
    fun parallel(): Type {
        val solver = when {
            method == "simple" -> ParallelSGDSolver(learningRate, workingThreads, stepDecay)
            method.startsWith(CLUSTER_METHOD_PREFIX) -> ClusterParallelSGDSolver(learningRate, workingThreads, stepDecay, method.substring(CLUSTER_METHOD_PREFIX.length).toInt())
            else -> error("Unknown method $method")
        }
        val result = solver.solve(loss.first, loss.second, targetLoss)
        return loss.second.loss(result.w)
    }
}
