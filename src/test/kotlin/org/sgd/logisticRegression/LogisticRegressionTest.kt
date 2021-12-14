package org.sgd.logisticRegression

import benchmark.Profiler
import kotlinx.smartbench.benchmark.*
import kotlinx.smartbench.declarative.Operation
import kotlinx.smartbench.graphic.PlotConfiguration
import kotlinx.smartbench.graphic.Scaling
import kotlinx.smartbench.graphic.ValueAxis
import org.junit.jupiter.api.Test
import org.sgd.*
import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

private const val AVERAGE_LOSS_METRIC = "average loss"
private const val SPEEDUP_METRIC = "speedup"
private const val CLUSTER_METHOD_PREFIX = "cluster-"

private const val DATASET = "rcv1"

val baseDir = File("../datasets").let {
    if (File(it, DATASET).exists()) it
    else File("/home/maksim.zuev/datasets")
}

private val dataset by lazy { loadBinaryDataSet(File(baseDir, DATASET), File(baseDir, "$DATASET.t")) }
private val features by lazy { dataset.first.points.asSequence().plus(dataset.second.points).maxOf { it.indices.maxOrNull() ?: 0 } }

class RunRegressionTask(
    val method: String, val learningRate: Type, val stepDecay: Type, val workingThreads: Int, val targetLoss: Type
) : Benchmark() {
    private val trainLoss = DataSetLoss(dataset.first, LogisticRegressionLoss())
    private val testLoss = DataSetLoss(dataset.second, LogisticRegressionLoss())

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
    private val trainLoss = DataSetLoss(dataset.first, LogisticRegressionLoss())
    private val testLoss = DataSetLoss(dataset.second, LogisticRegressionLoss())
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
            param(SequentialRegressionTask::learningRate, 0.5.toType())
            param(SequentialRegressionTask::stepDecay, 0.8.toType())
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
        val trainLoss = DataSetLoss(dataset.first, LogisticRegressionLoss())
        val testLoss = DataSetLoss(dataset.second, LogisticRegressionLoss())
        val solver = ClusterParallelSGDSolver(0.5.toType(), 128, 0.8.toType(), 32)
        val result = solver.solve(trainLoss, testLoss, TypeArray(features + 1), 0.025.toType())
        println(testLoss.loss(result.w))
    }

    @Test
    fun solverCompareWithProfiler() = solverCompareTest({
        attachProfiler(Profiler.LINUX_PEF_NORM_PROFILER)
    }) {
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
        for (p: String in logSequence(numaConfig.values.maxOf { it.size }).map { "$CLUSTER_METHOD_PREFIX$it" }) {
            iterationResults.entries.filter {
                it.key.params[RunRegressionTask::method.name]!!.param == p
            }.forEach {
                it.value.metrics["store_misses, %"] = it.value.metrics["LLC-store-misses, #/op"]!!.toDouble() / it.value.metrics["LLC-stores, #/op"]!!.toDouble()
                it.value.metrics["load_misses, %"] = it.value.metrics["LLC-load-misses, #/op"]!!.toDouble() / it.value.metrics["LLC-loads, #/op"]!!.toDouble()
            }
        }
        plot(xParameter = RunRegressionTask::workingThreads) {
            configure("store_misses%")
            valueAxis(ValueAxis.CustomMetric("store_misses, %"))
        }
        plot(xParameter = RunRegressionTask::workingThreads) {
            configure("load_misses%")
            valueAxis(ValueAxis.CustomMetric("load_misses, %"))
        }
    }

    private fun PlotConfiguration<RunRegressionTask>.configure(name: String) {
        filename("results/$name.png")
        xScaling(Scaling.LOGARITHMIC)
        useErrorBars(true)
    }

    @Test
    fun solverCompare() = solverCompareTest()

    private fun solverCompareTest(
        configureBenchmark: BenchmarkConfiguration<RunRegressionTask>.() -> Unit = {},
        plotExtra: BenchmarkResults<RunRegressionTask>.() -> Unit = {}
    ) {
        val threadsPerCluster = logSequence(numaConfig.values.maxOf { it.size }).map { "$CLUSTER_METHOD_PREFIX$it" }
        val threads = logSequence(Runtime.getRuntime().availableProcessors())
        runBenchmark<RunRegressionTask> {
            param(RunRegressionTask::method, threadsPerCluster)
            param(RunRegressionTask::learningRate, 0.5.toType())
            param(RunRegressionTask::stepDecay, 0.8.toType())
            param(RunRegressionTask::targetLoss, 0.024.toType())
            param(RunRegressionTask::workingThreads, threads)
            approximateBatchSize(50)
            measurementMode(MeasurementMode.AVERAGE_TIME, TimeUnit.SECONDS)
            configureBenchmark()
        }.run {
            File("results").run {
                mkdir()
                println(absolutePath)
            }

            plot(xParameter = RunRegressionTask::workingThreads) {
                configure("time")
            }
            plot(xParameter = RunRegressionTask::workingThreads) {
                for (p: String in threadsPerCluster) {
                    val oneThreadTime = iterationResults.entries.single {
                        val params = it.key.params
                        params[RunRegressionTask::method.name]!!.param == p &&
                            params[RunRegressionTask::workingThreads.name]!!.param == 1
                    }.value.resultValue(benchmarkConfiguration)
                    iterationResults.entries.filter {
                        it.key.params[RunRegressionTask::method.name]!!.param == p
                    }.forEach {
                        it.value.metrics[SPEEDUP_METRIC] = oneThreadTime / it.value.resultValue(benchmarkConfiguration)
                    }
                }
                configure(SPEEDUP_METRIC)
                valueAxis(ValueAxis.CustomMetric(SPEEDUP_METRIC))
            }
            plotExtra()
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

fun splitDataset() {
    val source = ""
    val train = "../datasets/xxx"
    val test = "../datasets/xxx.t"
    File(source).useLines {
        val strings = it.toMutableList()
        strings.shuffle()
        File(train).run {
            createNewFile()
            writeText(strings.subList(0, (strings.size * 0.8).toInt()).joinToString("\n"))
        }
        File(test).run {
            createNewFile()
            writeText(strings.subList((strings.size * 0.8).toInt(), strings.size).joinToString("\n"))
        }
    }
}
