package org.sgd

import kotlinx.atomicfu.AtomicIntArray
import kotlinx.smartbench.benchmark.Benchmark
import kotlinx.smartbench.benchmark.MeasurementMode
import kotlinx.smartbench.benchmark.named
import kotlinx.smartbench.benchmark.runBenchmark
import kotlinx.smartbench.declarative.Operation
import kotlinx.smartbench.graphic.Scaling
import kotlinx.smartbench.graphic.ValueAxis
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.TimeUnit

class ShuffleBenchmark(val workingThreads: Int, val size: Int) : Benchmark() {
    val data = MutableList(1 shl size) { it }
    val status = AtomicIntArray(Runtime.getRuntime().availableProcessors())
    val barrier = CyclicBarrier(workingThreads)

    @Operation
    fun shuffle(): Int {
        if (workingThreads == 1) {
            data.shuffle()
            return data[0]
        }
        List(workingThreads) {
            Thread { parallelShuffleTask(data, it, workingThreads, status) }
        }
            .onEach { it.start() }
            .onEach { it.join() }
        return data[0]
    }
}


internal class ShuffleTest {

    @Test
    fun shuffle() {
        val list = (1..10).toList()
        val c = list.toMutableList()
        repeat(100) {
            c.shuffle()
            assertEquals(list.size, c.size)
            for (e in list) {
                assertTrue(c.contains(e))
            }
        }
    }

    @Test
    fun shuffleBenchmark() {
        val threads = logSequence(Runtime.getRuntime().availableProcessors())
        val sizes = 19..24

        runBenchmark<ShuffleBenchmark> {
            param(ShuffleBenchmark::size, sizes.map { it named "2^$it" })
            param(ShuffleBenchmark::workingThreads, threads.map { it named "$it" })
            approximateBatchSize(50)
            measurementMode(MeasurementMode.AVERAGE_TIME, TimeUnit.MILLISECONDS)
        }.run {
            for (size in sizes) {
                val oneThreadTime = iterationResults.entries.single {
                    val params = it.key.params
                    params[ShuffleBenchmark::size.name]!!.param == size &&
                        params[ShuffleBenchmark::workingThreads.name]!!.param == 1
                }.value.resultValue(benchmarkConfiguration)
                iterationResults.entries.filter {
                    it.key.params[ShuffleBenchmark::size.name]!!.param == size
                }.forEach {
                    it.value.metrics[SPEEDUP_METRIC] = oneThreadTime / it.value.resultValue(benchmarkConfiguration)
                }
            }
            plot(ShuffleBenchmark::size) {
                filename("speedup.png")
                xScaling(Scaling.LOGARITHMIC)
                valueAxis(ValueAxis.CustomMetric(SPEEDUP_METRIC))
            }
            plot(ShuffleBenchmark::size) {
                filename("time.png")
                xScaling(Scaling.LOGARITHMIC)
                useErrorBars(true)
            }
        }
    }
}
