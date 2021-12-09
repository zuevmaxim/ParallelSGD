package org.sgd

import benchmark.Profiler
import kotlinx.smartbench.benchmark.*
import kotlinx.smartbench.declarative.Operation
import kotlinx.smartbench.graphic.PlotConfiguration
import kotlinx.smartbench.graphic.Scaling
import kotlinx.smartbench.graphic.ValueAxis
import org.junit.jupiter.api.Test
import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt
import kotlin.math.sqrt

private const val AVERAGE_LOSS_METRIC = "average loss"
private const val CLUSTER_METHOD_PREFIX = "cluster-"

val baseDir =
    File("/home/maksim.zuev/datasets")
//    File(".")

const val DATASET = "rcv1"

val train by lazy { loadDataSet(File(baseDir, DATASET)) }
val test by lazy { loadDataSet(File(baseDir, "$DATASET.t")) }
val features by lazy { test.points.asSequence().plus(train.points).maxOf { it.indices.maxOrNull() ?: 0 } }

class RunRegressionTask(
    val method: String, val learningRate: Type, val stepDecay: Type, val workingThreads: Int, val targetLoss: Type
) : Benchmark() {
    private val trainLoss = LinearRegressionLoss(train)
    private val testLoss = LinearRegressionLoss(test)

    @Operation
    fun run(): Type {
        val solver = when {
            method == "simple" -> ParallelSGDSolver(learningRate, workingThreads, stepDecay)
            method.startsWith(CLUSTER_METHOD_PREFIX) -> ClusterParallelSGDSolver(learningRate, workingThreads, stepDecay, method.substring(CLUSTER_METHOD_PREFIX.length).toInt())
            else -> error("Unknown method $method")
        }
        val result = solver.solve(trainLoss, testLoss, TypeArray(features + 1), targetLoss)
        return testLoss.loss(result.w)
    }
}

class SequentialRegressionTask(
    val learningRate: Type, val stepDecay: Type, val iterations: Int
) : Benchmark() {
    val trainLoss = LinearRegressionLoss(train)
    val testLoss = LinearRegressionLoss(test)
    var loss = BenchmarkCounter()
    val counter = BenchmarkCounter()

    @Operation
    fun run(): Type {
        val solver = SequentialSGDSolver(iterations, learningRate, stepDecay)
        val result = solver.solve(trainLoss, testLoss, TypeArray(features + 1), ZERO)
        return testLoss.loss(result.w).also {
            val scale = 1e3.toLong()
            loss.inc((it * scale).toLong())
            counter.inc(scale)
        }
    }
}

class LinearRegressionTest {

    @Test
    fun testNumaConfig() {
        for ((node, cpus) in numaConfig) {
            println("$node $cpus")
        }
    }

    @Test
    fun sequentialSolver() {
        runBenchmark<SequentialRegressionTask> {
            param(SequentialRegressionTask::learningRate, 0.5)
            param(SequentialRegressionTask::stepDecay, 0.8)
            param(SequentialRegressionTask::iterations, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            approximateBatchSize(10)
            measurementMode(MeasurementMode.AVERAGE_TIME, TimeUnit.SECONDS)
            metric(AVERAGE_LOSS_METRIC) { loss.value / counter.value.toDouble() }
            benchmarkMode(BenchmarkMode.INPLACE)
        }.run {
            plot(xParameter = SequentialRegressionTask::iterations) {
                valueAxis(ValueAxis.CustomMetric(AVERAGE_LOSS_METRIC))
            }
        }
    }

    @Test
    fun run() {
        val trainLoss = LinearRegressionLoss(train)
        val testLoss = LinearRegressionLoss(test)
        val solver = ClusterParallelSGDSolver(0.5f, 128, 0.8f, 32)
        val result = solver.solve(trainLoss, testLoss, TypeArray(features + 1), 0.028f)
        testLoss.loss(result.w)
    }

    @Test
    fun solverCompare() {
        runBenchmark<RunRegressionTask> {
            param(RunRegressionTask::method, logSequence(numaConfig.values.maxOf { it.size }).map { "$CLUSTER_METHOD_PREFIX$it" })
            param(RunRegressionTask::learningRate, 0.5f)
            param(RunRegressionTask::stepDecay, 0.8f)
            param(RunRegressionTask::targetLoss, 0.025f)
            param(RunRegressionTask::workingThreads, logSequence(Runtime.getRuntime().availableProcessors(), sqrt(2.0)))
            approximateBatchSize(30)
            measurementMode(MeasurementMode.AVERAGE_TIME, TimeUnit.SECONDS)
            attachProfiler(Profiler.LINUX_PEF_NORM_PROFILER)
        }.run {
            File("results").run {
                mkdir()
                println(absolutePath)
            }
            fun PlotConfiguration<RunRegressionTask>.configure(name: String) {
                filename("results/$name.png")
                xScaling(Scaling.LOGARITHMIC)
                useErrorBars(true)
            }

            plot(xParameter = RunRegressionTask::workingThreads) {
                configure("time")
            }
            plot(xParameter = RunRegressionTask::workingThreads) {
                configure("LLC_store_misses")
                valueAxis(ValueAxis.LLC_store_misses)
            }
            plot(xParameter = RunRegressionTask::workingThreads) {
                configure("LLC_stores")
                valueAxis(ValueAxis.LLC_stores)
            }
            plot(xParameter = RunRegressionTask::workingThreads) {
                configure("LLC_loads")
                valueAxis(ValueAxis.LLC_loads)
            }
            plot(xParameter = RunRegressionTask::workingThreads) {
                configure("LLC_load_misses")
                valueAxis(ValueAxis.LLC_load_misses)
            }
        }
    }
}

//fun getInterestingThreads(): List<Int> {
//    val result = mutableListOf<Int>()
//    val numaNodes = numaConfig.mapValues { it.value.size }
//    for (i in numaNodes.keys.sorted()) {
//        val threads = numaNodes[i]!!
//        val last = result.lastOrNull() ?: 0
//        result.addAll(logSequence(threads/*, sqrt(2.0)*/).map { it + last })
//    }
//    return result
//}

fun logSequence(maxValue: Int, step: Double = 2.0): List<Int> {
    var value = maxValue.toDouble()
    val result = hashSetOf<Int>()
    while (value >= 1.0) {
        result.add(value.roundToInt())
        value /= step
    }
    return result.toList().sorted()
}
