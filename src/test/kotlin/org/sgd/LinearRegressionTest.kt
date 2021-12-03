package org.sgd

import jetbrains.letsPlot.export.ggsave
import jetbrains.letsPlot.geom.geomLine
import jetbrains.letsPlot.geom.geomPoint
import jetbrains.letsPlot.letsPlot
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.math.roundToInt
import kotlin.math.sqrt

val baseDir = File("/home/maksim.zuev/datasets")
const val DATASET = "epsilon_normalized"

class LinearRegressionTest {

    @Test
    fun testNumaConfig() {
        for ((node, cpus) in numaConfig) {
            println("$node $cpus")
        }
    }

    @Test
    fun sequentialSolver() {
        val testDataSet = loadDataSet(File(baseDir, "$DATASET.t"))
        val trainDataSet = loadDataSet(File(baseDir, DATASET))
        print("DataSet loaded")
        val features = testDataSet.points.asSequence().plus(trainDataSet.points).maxOf { it.indices.maxOrNull() ?: 0 } + 1
        println("Features $features")
        val loss = LinearRegressionLoss(trainDataSet)
        val testLoss = LinearRegressionLoss(testDataSet)

        var bestIterations = 0
        var bestLearningRate = 0.0
        var bestStepDecay = 0.0
        var bestLoss = Double.MAX_VALUE
        for (iterations in listOf(1, 2, 5, 10, 20)) {
            for (learningRate in 1..100 step 10) {
                for (stepDecay in 80..100 step 10) {
                    val currentLoss = generateSequence {
                        System.gc()
                        val solver = SequentialSGDSolver(iterations, learningRate / 100.0, stepDecay / 100.0)
                        val result = solver.solve(loss, testLoss, DoubleArray(features + 1), 0.0)
                        testLoss.loss(result.w)
                    }.take(3).average()

                    println("$iterations $learningRate $stepDecay $currentLoss")
                    if (currentLoss < bestLoss) {
                        bestIterations = iterations
                        bestLoss = currentLoss
                        bestLearningRate = learningRate / 100.0
                        bestStepDecay = stepDecay / 100.0
                    }
                }
            }
        }

        println("$bestIterations $bestLearningRate $bestStepDecay")
        val solver = SequentialSGDSolver(bestIterations, bestLearningRate, bestStepDecay)
        val result = solver.solve(loss, testLoss, DoubleArray(features + 1), 0.0)
        val totalTimeMs = result.timeNsToWeights.keys.maxOrNull()!! / 1e6
        val (tp, fp) = testLoss.precision(result.w)
        val lv = testLoss.loss(result.w)
        println("time: $totalTimeMs ms; $tp $fp $lv")
        val timeMs = mutableListOf<Double>()
        val truePrecision = mutableListOf<Double>()
        val falsePrecision = mutableListOf<Double>()
        val losss = mutableListOf<Double>()
        val solverNames = mutableListOf<String>()
        for ((timeNs, w) in result.timeNsToWeights) {
            val (tp, fp) = testLoss.precision(w)
            timeMs.add(timeNs / 1e6)
            truePrecision.add(tp)
            falsePrecision.add(fp)
            solverNames.add("sequential")
            losss.add(testLoss.loss(w))
        }
        plotPrecision(timeMs, solverNames, truePrecision, "true")
        plotPrecision(timeMs, solverNames, falsePrecision, "false")
        plotPrecision(timeMs, solverNames, losss, "loss")
    }

    @Test
    fun solverCompare() {
        val learningRate = 0.1
        val stepDecay = 0.85
        val threads = mutableListOf<Int>().apply {
            var t = Runtime.getRuntime().availableProcessors().toDouble()
            while (t.toInt() > 0) {
                add(t.roundToInt())
                t /= sqrt(2.0)
            }
        }.reversed()
        val solvers = threads.map { i -> Triple("simple", i, ParallelSGDSolver(learningRate, i, stepDecay)) } +
            threads.map { i -> Triple("cluster", i, ClusterParallelSGDSolver(learningRate, i, stepDecay)) }

        val testDataSet = loadDataSet(File(baseDir, "$DATASET.t"))
        val trainDataSet = loadDataSet(File(baseDir, DATASET))
        val features = testDataSet.points.asSequence().plus(trainDataSet.points).maxOf { it.indices.maxOrNull() ?: 0 } + 1
        val loss = LinearRegressionLoss(trainDataSet)
        val testLoss = LinearRegressionLoss(testDataSet)
        val log = File(baseDir, "$DATASET.txt").writer().buffered()

        for ((name, threads, solver) in solvers) {
            val runs = 5
            val (timeMs, tp, mse) = generateSequence {
                System.gc()
                val result = solver.solve(loss, testLoss, DoubleArray(features + 1), 0.1)
                val lastResult = result.timeNsToWeights.maxByOrNull { it.key }!!
                val totalTimeMs = lastResult.key / 1e6
                val (tp, _) = testLoss.precision(lastResult.value)
                val mse = testLoss.loss(lastResult.value)
                println("$name, $threads, $totalTimeMs, $tp, $mse")
                log.write("$name, $threads, $totalTimeMs, $tp, $mse\n")
                Triple(totalTimeMs, tp, mse)
            }.drop(2).take(runs).fold(Triple(0.0, 0.0, 0.0)) { acc, x -> acc + x } / runs

            println("===$name $threads time: $timeMs ms; $tp $mse")
        }
        log.close()
    }

    private fun plotPrecision(timeMs: List<Double>, solverNames: List<String>, precision: List<Double>, name: String) {
        val data = mutableMapOf(
            "time(ms)" to timeMs,
            "precision" to precision,
            "solver" to solverNames
        )
        var plot = letsPlot(data) {
            x = "time(ms)"
            y = "precision"
            color = "solver"
        }
        plot += geomLine()
        plot += geomPoint()
        ggsave(plot, "$name.png")
    }
}


private fun loadDataSet(file: File): DataSet {
    val points = mutableListOf<DataPoint>()
    file.useLines { lines ->
        lines.forEach { line ->
            val parts = line.trim().split(" ")
            val y = if (parts[0].toInt() == 1) 1.0 else 0.0
            val (indices, values) = parts.drop(1).map { val (id, count) = it.split(':'); id.toInt() to count.toDouble() }.unzip()
            points.add(DataPoint(indices.toIntArray(), values.toDoubleArray(), y))
        }
    }
    return DataSet(points)
}

operator fun Triple<Double, Double, Double>.plus(other: Triple<Double, Double, Double>) = Triple(first + other.first, second + other.second, third + other.third)
operator fun Triple<Double, Double, Double>.div(other: Number) = Triple(first / other.toDouble(), second / other.toDouble(), third / other.toDouble())
