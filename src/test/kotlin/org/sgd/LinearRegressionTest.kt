package org.sgd

import jetbrains.letsPlot.export.ggsave
import jetbrains.letsPlot.geom.geomLine
import jetbrains.letsPlot.geom.geomPoint
import jetbrains.letsPlot.letsPlot
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt


class LinearRegressionTest {

    @Test
    fun testNumaConfig() {
        for ((node, cpus) in numaConfig) {
            println("$node $cpus")
        }
    }

    @Test
    fun solverCompare() {
        val timeMs = mutableListOf<Double>()
        val truePrecision = mutableListOf<Double>()
        val falsePrecision = mutableListOf<Double>()
        val solverNames = mutableListOf<String>()

        val threadsList = mutableListOf<Int>()
        val solversList = mutableListOf<String>()
        val maxTimeList = mutableListOf<Double>()

        val iterationsNumber = 200

        val learningRate = 0.5
        val stepDecay = 0.95
        val threads = mutableListOf<Int>() .apply {
            var t = Runtime.getRuntime().availableProcessors().toDouble()
            while (t.toInt() > 0) {
                add(t.roundToInt())
                t /= sqrt(2.0)
            }
        }.reversed()
        val solvers = listOf(Triple("sequential", 1, SequentialSGDSolver(iterationsNumber, learningRate, stepDecay))) +
            threads.map { i -> Triple("simple", i, ParallelSGDSolver(iterationsNumber, learningRate, i, stepDecay)) } +
            threads.map { i -> Triple("cluster", i, ClusterParallelSGDSolver(iterationsNumber, learningRate, i, stepDecay)) }

        val testDataSet = loadDataSet(File("w8a.t"))
        val trainDataSet = loadDataSet(File("w8a"))
        val features = testDataSet.points.asSequence().plus(trainDataSet.points).maxOf { it.indices.maxOrNull() ?: 0 } + 1
        val loss = LinearRegressionLoss(trainDataSet)
        val testLoss = LinearRegressionLoss(testDataSet)

        for ((name, threads, solver) in solvers) {
            val result = solver.solve(loss, DoubleArray(features + 1))

            val totalTimeMs = result.timeNsToWeights.keys.maxOrNull()!! / 1e6
            val (tp, fp) = testLoss.precision(result.w)
            println("$name $threads time: $totalTimeMs ms; $tp $fp")

            threadsList.add(threads)
            solversList.add(name)
            maxTimeList.add(totalTimeMs)

            for ((timeNs, lv) in result.timeNsToWeights.mapValues { testLoss.precision(it.value) }) {
                timeMs.add(timeNs / 1e6)
                truePrecision.add(lv.first)
                falsePrecision.add(lv.second)
                solverNames.add(name + threads)
            }
        }

        plotPrecision(timeMs, solverNames, truePrecision, "true")
        plotPrecision(timeMs, solverNames, falsePrecision, "false")


        val dataTime = mutableMapOf(
            "time(ms)" to maxTimeList,
            "threads" to threadsList,
            "solver" to solversList
        )
        var plotTime = letsPlot(dataTime) {
            x = "threads"
            y = "time(ms)"
            color = "solver"
        }
        plotTime += geomLine()
        plotTime += geomPoint()
        ggsave(plotTime, "time.png")
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

class LinearRegressionLoss(private val dataSet: DataSet) : DataSetLoss(dataSet) {
    override fun pointLoss(p: DataPoint) = LinearRegressionPointLoss(p)

    fun precision(w: Weights): Pair<Double, Double> {
        var truePositive = 0
        var falsePositive = 0
        var trueNegative = 0
        var falseNegative = 0
        for (point in dataSet.points) {
            val prediction = if (dot(w, point) >= 0) 1.0 else 0.0
            if (prediction == 1.0) {
                if (prediction == point.y) truePositive ++ else falsePositive++
            } else {
                if (prediction == point.y) trueNegative++ else falseNegative++
            }
        }
        return truePositive.toDouble() / (truePositive + falseNegative) to trueNegative.toDouble() / (trueNegative + falsePositive)
    }
}

class LinearRegressionPointLoss(private val p: DataPoint) : DataPointLoss {
    override fun loss(w: Weights) = (p.y - if (dot(w, p) >= 0.0) 1.0 else 0.0).pow(2)

    override fun gradientStep(w: Weights, learningRate: Double) {
        val indices = p.indices
        val xs = p.xValues
        val e = exp(-dot(w, p))
        repeat(indices.size) { i ->
            w[indices[i]] -= learningRate * xs[i] * (1 / (1 + e) - p.y)
        }
        w[w.size - 1] -= learningRate * (1 / (1 + e) - p.y)
    }
}

private fun dot(w: Weights, p:DataPoint): Double {
    var s = w[w.size - 1]
    val indices = p.indices
    val xs = p.xValues
    repeat(indices.size) { i ->
        s += xs[i] * w[indices[i]]
    }
    return s
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


