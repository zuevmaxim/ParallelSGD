package org.sgd

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
            param(SequentialRegressionTask::learningRate, 0.06)
            param(SequentialRegressionTask::stepDecay, 1.0)
            param(SequentialRegressionTask::iterations, 100)
            approximateBatchSize(10)
            measurementMode(MeasurementMode.AVERAGE_TIME, TimeUnit.SECONDS)
            metric(AVERAGE_LOSS_METRIC) { loss.value / counter.value.toDouble() }
        }.run {
            plot(xParameter = SequentialRegressionTask::learningRate) {
                valueAxis(ValueAxis.CustomMetric(AVERAGE_LOSS_METRIC))
            }
        }
    }

    @Test
    fun solverCompare() {
        runBenchmark<RunRegressionTask> {
            param(RunRegressionTask::method, "simple", "cluster")
            param(RunRegressionTask::learningRate, 0.06)
            param(RunRegressionTask::stepDecay, 1.0)
            param(RunRegressionTask::targetLoss, 0.013)
            param(RunRegressionTask::workingThreads, logSequence(Runtime.getRuntime().availableProcessors()))
            approximateBatchSize(5)
            measurementMode(MeasurementMode.AVERAGE_TIME, TimeUnit.SECONDS)
//            attachProfiler(Profiler.LINUX_PEF_NORM_PROFILER)
        }.run {
            File("results").run {
                mkdir()
                println(absolutePath)
            }
            val configure: PlotConfiguration<RunRegressionTask>.(name: String) -> Unit = {
                useErrorBars(true)
                yScaling(Scaling.LOGARITHMIC)
                filename("results/$name.png")
            }

            plot(xParameter = RunRegressionTask::workingThreads) {
                configure("time")
            }
//                plot(xParameter = RunRegressionTask::workingThreads) {
//                    useErrorBars(true)
//                    valueAxis(ValueAxis.LLC_stores)
//                    filename("results/LLC_stores.png")
//                }
//                plot(xParameter = RunRegressionTask::workingThreads) {
//                    useErrorBars(true)
//                    valueAxis(ValueAxis.LLC_loads)
//                    filename("results/LLC_loads.png")
//                }
//                plot(xParameter = RunRegressionTask::workingThreads) {
//                    useErrorBars(true)
//                    valueAxis(ValueAxis.LLC_load_misses)
//                    filename("results/LLC_load_misses.png")
//                }
//                plot(xParameter = RunRegressionTask::workingThreads) {
//                    useErrorBars(true)
//                    valueAxis(ValueAxis.LLC_store_misses)
//                    filename("results/LLC_store_misses.png")
//                }
        }
    }
}

fun logSequence(maxValue: Int, step: Double = 2.0): List<Int> {
    var value = maxValue.toDouble()
    val result = hashSetOf<Int>()
    while (value >= 1.0) {
        result.add(value.roundToInt())
        value /= step
    }
    return result.toList().sorted()
}
