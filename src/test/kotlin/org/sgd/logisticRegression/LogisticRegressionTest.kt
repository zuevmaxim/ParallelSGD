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

private const val DATASET = "rcv1"
private val dataset by lazy { loadBinaryDataSet(File(baseDir, DATASET), File(baseDir, "$DATASET.t")) }
private val features by lazy { dataset.first.points.asSequence().plus(dataset.second.points).maxOf { it.indices.maxOrNull() ?: 0 } }

class RunRegressionTask(
    val method: String, val learningRate: Type, val stepDecay: Type, val workingThreads: Int, val targetLoss: Type
) : Benchmark() {
    private val trainLoss = LogisticRegressionModel(dataset.first, features)
    private val testLoss = LogisticRegressionModel(dataset.second, features)

    @Operation
    fun run(): Type {
        val solver = when {
            method == "simple" -> ParallelSGDSolver(learningRate, workingThreads, stepDecay)
            method.startsWith(CLUSTER_METHOD_PREFIX) -> ClusterParallelSGDSolver(learningRate, workingThreads, stepDecay, method.substring(CLUSTER_METHOD_PREFIX.length).toInt())
            else -> error("Unknown method $method")
        }
        val result = solver.solve(trainLoss, testLoss, targetLoss)
        return testLoss.loss(result.w)
    }
}

class SequentialRegressionTask(
    val learningRate: Type, val stepDecay: Type, val iterations: Int
) : Benchmark() {
    private val trainLoss = LogisticRegressionModel(dataset.first, features)
    private val testLoss = LogisticRegressionModel(dataset.second, features)
    var loss = BenchmarkCounter()
    val counter = BenchmarkCounter()

    @Operation
    fun run(): Type {
        val solver = SequentialSGDSolver(iterations, learningRate, stepDecay)
        val result = solver.solve(trainLoss, testLoss, ZERO)
        return testLoss.loss(result.w).also {
            val scale = 1e3.toLong()
            loss.inc((it * scale).toLong())
            counter.inc(scale)
        }
    }
}

class LinearRegressionTest {

    @Test
    fun sequentialSolver() {
        val p = params[DATASET]!!
        runBenchmark<SequentialRegressionTask> {
            param(SequentialRegressionTask::learningRate, p["learningRate"])
            param(SequentialRegressionTask::stepDecay, p["stepDecay"])
            param(SequentialRegressionTask::iterations, 1..20)
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
        val trainLoss = LogisticRegressionModel(dataset.first, features)
        val testLoss = LogisticRegressionModel(dataset.second, features)
        val p = params[DATASET]!!
        val solver = ClusterParallelSGDSolver(p["learningRate"]!!, 128, p["stepDecay"]!!, 32)
        val result = solver.solve(trainLoss, testLoss, p["targetLoss"]!!)
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
        val p = params[DATASET]!!
        runBenchmark<RunRegressionTask> {
            param(RunRegressionTask::method, threadsPerCluster)
            param(RunRegressionTask::learningRate, p["learningRate"])
            param(RunRegressionTask::stepDecay, p["stepDecay"])
            param(RunRegressionTask::targetLoss, p["targetLoss"])
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
