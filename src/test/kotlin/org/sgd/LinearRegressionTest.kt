package org.sgd

import benchmark.Profiler
import kotlinx.smartbench.benchmark.Benchmark
import kotlinx.smartbench.benchmark.MeasurementMode
import kotlinx.smartbench.benchmark.param
import kotlinx.smartbench.benchmark.runBenchmark
import kotlinx.smartbench.declarative.Operation
import kotlinx.smartbench.graphic.PlotConfiguration
import kotlinx.smartbench.graphic.Scaling
import kotlinx.smartbench.graphic.ValueAxis
import org.apache.batik.svggen.font.table.Table.name
import org.junit.jupiter.api.Test
import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt
import kotlin.math.sqrt

const val AVERAGE_LOSS_METRIC = "average loss"

val baseDir =
    File("/home/maksim.zuev/datasets")
//    File(".")

const val DATASET = "rcv1"

val train by lazy { loadDataSet(File(baseDir, DATASET)) }
val test by lazy { loadDataSet(File(baseDir, "$DATASET.t")) }
val features by lazy { test.points.asSequence().plus(train.points).maxOf { it.indices.maxOrNull() ?: 0 } }

class RunRegressionTask(
    val method: String, val learningRate: Double, val stepDecay: Double, val workingThreads: Int, val targetLoss: Double
) : Benchmark() {
    private val trainLoss = LinearRegressionLoss(train)
    private val testLoss = LinearRegressionLoss(test)

    @Operation
    fun run(): LossValue {
        val solver = when (method) {
            "simple" -> ParallelSGDSolver(learningRate, workingThreads, stepDecay)
            "cluster" -> ClusterParallelSGDSolver(learningRate, workingThreads, stepDecay)
            else -> error("Unknown method $method")
        }
        val result = solver.solve(trainLoss, testLoss, DoubleArray(features + 1), targetLoss)
        return testLoss.loss(result.w)
    }
}

class SequentialRegressionTask(
    val learningRate: Double, val stepDecay: Double, val iterations: Int
) : Benchmark() {
    val trainLoss = LinearRegressionLoss(train)
    val testLoss = LinearRegressionLoss(test)
    var loss = BenchmarkCounter()
    val counter = BenchmarkCounter()

    @Operation
    fun run(): LossValue {
        val solver = SequentialSGDSolver(iterations, learningRate, stepDecay)
        val result = solver.solve(trainLoss, testLoss, DoubleArray(features + 1), 0.0)
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
            approximateBatchSize(5)
            measurementMode(MeasurementMode.AVERAGE_TIME, TimeUnit.SECONDS)
            metric(AVERAGE_LOSS_METRIC) { loss.value / counter.value.toDouble() }
        }.run {
            plot(xParameter = SequentialRegressionTask::iterations) {
                valueAxis(ValueAxis.CustomMetric(AVERAGE_LOSS_METRIC))
            }
        }
    }

    @Test
    fun solverCompare() {
        runBenchmark<RunRegressionTask> {
            param(RunRegressionTask::method, "simple", "cluster")
            param(RunRegressionTask::learningRate, 0.5)
            param(RunRegressionTask::stepDecay, 0.8)
            param(RunRegressionTask::targetLoss, 0.025)
            param(RunRegressionTask::workingThreads, logSequence(Runtime.getRuntime().availableProcessors(), sqrt(2.0)))
            approximateBatchSize(10)
            measurementMode(MeasurementMode.AVERAGE_TIME, TimeUnit.SECONDS)
            attachProfiler(Profiler.LINUX_PEF_NORM_PROFILER)
        }.run {
            File("results").run {
                mkdir()
                println(absolutePath)
            }
            val configure: PlotConfiguration<RunRegressionTask>.(name: String) -> Unit = {
                useErrorBars(true)
                filename("results/$name.png")
                xScaling(Scaling.LOGARITHMIC)
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
