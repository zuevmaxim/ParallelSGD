package org.sgd

import benchmark.Profiler
import kotlinx.smartbench.benchmark.*
import kotlinx.smartbench.declarative.Operation
import kotlinx.smartbench.graphic.PlotConfiguration
import kotlinx.smartbench.graphic.Scaling
import kotlinx.smartbench.graphic.ValueAxis
import org.junit.jupiter.api.Test
import java.util.concurrent.TimeUnit

class LogisticRegressionTest {
    private val DATASET = "rcv1"

    @Test
    fun sequentialSolver() = runSequentialBenchmark(DATASET)

    @Test
    fun solverCompareWithProfiler() = runParallelBenchmarkWithProfiler(DATASET)

    @Test
    fun solverCompare() = runParallelBenchmark(DATASET)

    @Test
    fun run() {
        val dataset = DATASET
        val loss = models[dataset]!!()
        val p = params[dataset]!!
        val solver = ClusterParallelSGDSolver(p["learningRate"]!!, 128, p["stepDecay"]!!, 32)
        val result = solver.solve(loss.first, loss.second, p["targetLoss"]!!)
        println(loss.second.loss(result.w))
    }

    @Test
    fun runAllDatasets() {
        for (dataset in params.keys) {
            runParallelBenchmark(dataset)
        }
    }
}

class RunRegressionTask(
    val dataset: String,
    val method: String,
    val learningRate: Type,
    val stepDecay: Type,
    val workingThreads: Int,
    val targetLoss: Type
) : Benchmark() {
    private val loss = models[dataset]!!()

    @Operation
    fun run(): Type {
        val solver = when {
            method == "simple" -> ParallelSGDSolver(learningRate, workingThreads, stepDecay)
            method.startsWith(CLUSTER_METHOD_PREFIX) -> ClusterParallelSGDSolver(learningRate, workingThreads, stepDecay, method.substring(CLUSTER_METHOD_PREFIX.length).toInt())
            else -> error("Unknown method $method")
        }
        val result = solver.solve(loss.first, loss.second, targetLoss)
        return loss.second.loss(result.w)
    }
}

class SequentialRegressionTask(
    val dataset: String,
    val learningRate: Type,
    val stepDecay: Type,
    val iterations: Int
) : Benchmark() {
    private val lossModel = models[dataset]!!()
    var loss = BenchmarkCounter()
    val counter = BenchmarkCounter()

    @Operation
    fun run(): Type {
        val solver = SequentialSGDSolver(iterations, learningRate, stepDecay)
        val result = solver.solve(lossModel.first, lossModel.second, ZERO)
        return lossModel.second.loss(result.w).also {
            val scale = 1e3.toLong()
            loss.inc((it * scale).toLong())
            counter.inc(scale)
        }
    }
}

private fun <T : Benchmark> PlotConfiguration<T>.commonConfigure(dataset: String, name: String, logarithmic: Boolean = true) {
    filename("results/$dataset-$name.png")
    if (logarithmic) {
        xScaling(Scaling.LOGARITHMIC)
    }
    useErrorBars(true)
}

fun runSequentialBenchmark(dataset: String) {
    val p = params[dataset]!!
    runBenchmark<SequentialRegressionTask> {
        param(SequentialRegressionTask::dataset, dataset)
        param(SequentialRegressionTask::learningRate, p["learningRate"])
        param(SequentialRegressionTask::stepDecay, p["stepDecay"])
        param(SequentialRegressionTask::iterations, 1..10)
        approximateBatchSize(p["batch"]!!.toInt())
        measurementMode(MeasurementMode.AVERAGE_TIME, TimeUnit.SECONDS)
        metric(AVERAGE_LOSS_METRIC) { loss.value / counter.value.toDouble() }
        benchmarkMode(BenchmarkMode.INPLACE)
    }.run {
        plot(xParameter = SequentialRegressionTask::iterations) {
            commonConfigure(dataset, "time")
        }
        plot(xParameter = SequentialRegressionTask::iterations) {
            valueAxis(ValueAxis.CustomMetric(AVERAGE_LOSS_METRIC))
            commonConfigure(dataset, AVERAGE_LOSS_METRIC, false)
        }
    }
}

private fun runParallelBenchmark(
    dataset: String,
    configureBenchmark: BenchmarkConfiguration<RunRegressionTask>.() -> Unit = {},
    plotExtra: BenchmarkResults<RunRegressionTask>.() -> Unit = {}
) {
    val threadsPerCluster = logSequence(numaConfig.values.maxOf { it.size }).map { "$CLUSTER_METHOD_PREFIX$it" }
    val threads = logSequence(Runtime.getRuntime().availableProcessors())
    val p = params[dataset]!!
    runBenchmark<RunRegressionTask> {
        param(RunRegressionTask::dataset, dataset)
        param(RunRegressionTask::method, threadsPerCluster)
        param(RunRegressionTask::learningRate, p["learningRate"])
        param(RunRegressionTask::stepDecay, p["stepDecay"])
        param(RunRegressionTask::targetLoss, p["targetLoss"])
        param(RunRegressionTask::workingThreads, threads)
        approximateBatchSize(p["batch"]!!.toInt())
        measurementMode(MeasurementMode.AVERAGE_TIME, TimeUnit.SECONDS)
        configureBenchmark()
        output("results/$dataset.csv")
    }.run {
        plot(xParameter = RunRegressionTask::workingThreads) {
            commonConfigure(dataset, "time")
        }
//        plot(xParameter = RunRegressionTask::workingThreads) {
//            for (p: String in threadsPerCluster) {
//                val oneThreadTime = iterationResults.entries.single {
//                    val params = it.key.params
//                    params[RunRegressionTask::method.name]!!.param == p &&
//                        params[RunRegressionTask::workingThreads.name]!!.param == 1
//                }.value.resultValue(benchmarkConfiguration)
//                iterationResults.entries.filter {
//                    it.key.params[RunRegressionTask::method.name]!!.param == p
//                }.forEach {
//                    it.value.metrics[SPEEDUP_METRIC] = oneThreadTime / it.value.resultValue(benchmarkConfiguration)
//                }
//            }
//            commonConfigure(dataset, SPEEDUP_METRIC)
//            valueAxis(ValueAxis.CustomMetric(SPEEDUP_METRIC))
//        }
        plotExtra()
    }
}

fun runParallelBenchmarkWithProfiler(dataset: String) = runParallelBenchmark(dataset, {
    attachProfiler(Profiler.LINUX_PEF_NORM_PROFILER)
}) {
    plot(xParameter = RunRegressionTask::workingThreads) {
        commonConfigure(dataset, "LLC_store_misses")
        valueAxis(ValueAxis.LLC_store_misses)
    }
    plot(xParameter = RunRegressionTask::workingThreads) {
        commonConfigure(dataset, "LLC_stores")
        valueAxis(ValueAxis.LLC_stores)
    }
    plot(xParameter = RunRegressionTask::workingThreads) {
        commonConfigure(dataset, "LLC_loads")
        valueAxis(ValueAxis.LLC_loads)
    }
    plot(xParameter = RunRegressionTask::workingThreads) {
        commonConfigure(dataset, "LLC_load_misses")
        valueAxis(ValueAxis.LLC_load_misses)
    }
//    for (p: String in logSequence(numaConfig.values.maxOf { it.size }).map { "$CLUSTER_METHOD_PREFIX$it" }) {
//        iterationResults.entries.filter {
//            it.key.params[RunRegressionTask::method.name]!!.param == p
//        }.forEach {
//            it.value.metrics["store_misses, %"] = it.value.metrics["LLC-store-misses, #/op"]!!.toDouble() / it.value.metrics["LLC-stores, #/op"]!!.toDouble()
//            it.value.metrics["load_misses, %"] = it.value.metrics["LLC-load-misses, #/op"]!!.toDouble() / it.value.metrics["LLC-loads, #/op"]!!.toDouble()
//        }
//    }
//    plot(xParameter = RunRegressionTask::workingThreads) {
//        commonConfigure(dataset, "store_misses%")
//        valueAxis(ValueAxis.CustomMetric("store_misses, %"))
//    }
//    plot(xParameter = RunRegressionTask::workingThreads) {
//        commonConfigure(dataset, "load_misses%")
//        valueAxis(ValueAxis.CustomMetric("load_misses, %"))
//    }
}
