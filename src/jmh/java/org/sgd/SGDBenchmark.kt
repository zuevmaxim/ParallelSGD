package org.sgd

import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.OptionsBuilder
import java.io.File
import java.util.concurrent.TimeUnit


val baseDir = File(".")
const val DATASET = "w8a"

val train = loadDataSet(File(baseDir, DATASET))
val test = loadDataSet(File(baseDir, "$DATASET.t"))
val features = test.points.asSequence().plus(train.points).maxOf { it.indices.maxOrNull() ?: 0 }

@State(Scope.Benchmark)
@Measurement(iterations = 5, time = 10, timeUnit = TimeUnit.SECONDS)
@BenchmarkMode(Mode.AverageTime)
@Fork(1)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
open class SGDBenchmark {

    val trainLoss = LinearRegressionLoss(train)
    val testLoss = LinearRegressionLoss(test)

//    @Benchmark
    fun sequential(): LossValue {
        val iterations = 20
        val learningRate = 0.26
        val stepDecay = 1.0
        val solver = SequentialSGDSolver(iterations, learningRate, stepDecay)
        val result = solver.solve(trainLoss, testLoss, DoubleArray(features + 1), 0.0)
        return testLoss.loss(result.w)
    }

    @Param("0")
    var threads = 0

    @Benchmark
    fun parallel(): LossValue {
        val targetLoss = 0.02
        val learningRate = 0.26
        val stepDecay = 1.0
        val solver = ParallelSGDSolver(learningRate, threads, stepDecay)
        val result = solver.solve(trainLoss, testLoss, DoubleArray(features + 1), targetLoss)
        return testLoss.loss(result.w)
    }
}

fun main() {
    val opt = OptionsBuilder()
        .include(SGDBenchmark::class.qualifiedName + ".*")
        .param(SGDBenchmark::threads.name, *(1..Runtime.getRuntime().availableProcessors()).map { it.toString() }.toTypedArray())
        .build()

    Runner(opt).run()
}
