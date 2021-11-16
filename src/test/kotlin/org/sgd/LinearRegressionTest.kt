package org.sgd

import jetbrains.letsPlot.export.ggsave
import jetbrains.letsPlot.geom.geomLine
import jetbrains.letsPlot.geom.geomPoint
import jetbrains.letsPlot.letsPlot
import org.junit.Test
import kotlin.math.pow
import kotlin.random.Random


class LinearRegressionTest {

    @Test
    fun testNumaConfig() {
        println(numaConfig)
    }

    @Test
    fun solverCompare() {
        val timeS = mutableListOf<Double>()
        val trainLoss = mutableListOf<Double>()
        val solverNames = mutableListOf<String>()

        val threadsList = mutableListOf<Int>()
        val solversList = mutableListOf<String>()
        val maxTimeList = mutableListOf<Double>()

        val iterationsNumber = 100

        val learningRate = 0.00025
        val stepDecay = 0.99
        val threads = Runtime.getRuntime().availableProcessors()
        val solvers = List(threads) { i -> Triple("simple", i + 1, ParallelSGDSolver(iterationsNumber, learningRate / (i + 1), i + 1, stepDecay)) } +
            List(threads) { i -> Triple("cluster", i + 1, ClusterParallelSGDSolver(iterationsNumber, learningRate / (i + 1), i + 1, stepDecay)) }

        val features = 10000
        val size = 40000
        val (dataset, coefficients) = generateRandomDataSet(features, size)
        val (test, train) = dataset.split(0.5)
        val loss = LinearRegressionLoss(train)
        val testLoss = LinearRegressionLoss(test)

        println("Dataset loss: " + testLoss.loss(coefficients))
        for ((name, threads, solver) in solvers) {
            val result = solver.solve(loss, DoubleArray(features + 1))

            val totalTimeS = result.timeNsToWeights.keys.maxOrNull()!! / 1e9
            println("$name $threads time: $totalTimeS s; Test loss: " + testLoss.loss(result.w))

            threadsList.add(threads)
            solversList.add(name)
            maxTimeList.add(totalTimeS)

            for ((timeNs, lossValue) in result.timeNsToWeights.mapValues { testLoss.loss(it.value) }.filterValues { it < 2 }) {
                timeS.add(timeNs / 1e9)
                trainLoss.add(lossValue)
                solverNames.add(name)
            }
        }

        val data = mutableMapOf(
            "time(s)" to timeS,
            "train loss" to trainLoss,
            "solver" to solverNames
        )
        var plot = letsPlot(data) {
            x = "time(s)"
            y = "train loss"
            color = "solver"
        }
        plot += geomLine()
        plot += geomPoint()
        ggsave(plot, "compare.png")


        val dataTime = mutableMapOf(
            "time(s)" to maxTimeList,
            "threads" to threadsList,
            "solver" to solversList
        )
        var plotTime = letsPlot(dataTime) {
            x = "threads"
            y = "time(s)"
            color = "solver"
        }
        plotTime += geomLine()
        plotTime += geomPoint()
        ggsave(plotTime, "time.png")
    }
}

class LinearRegressionLoss(dataSet: DataSet) : DataSetLoss(dataSet) {
    override fun pointLoss(p: DataPoint) = LinearRegressionPointLoss(p)
}

class LinearRegressionPointLoss(private val p: DataPoint) : DataPointLoss {
    override fun loss(w: Weights) = (p.y - w.dot(p.x)).pow(2)

    override fun grad(w: Weights, buffer: Weights): Weights {
        require(buffer.size == w.size)
        val s = 2 * (w.dot(p.x) - p.y)
        repeat(buffer.size) { i ->
            buffer[i] = p.x[i] * s
        }
        return buffer
    }
}

private fun generateRandomDataSet(features: Int, size: Int): Pair<DataSet, DoubleArray> {
    val random = Random(42)
    val coefficients = Weights(features + 1) { random.nextDouble(-10.0, 10.0) }
    fun generateFeatures() = Features(features + 1) { i -> if (i == 0) 1.0 else random.nextDouble(1.0) }
    fun y(x: Features) = coefficients.dot(x) + random.nextDouble(-1.0, 1.0)
    return DataSet(Array(size) {
        val f = generateFeatures()
        DataPoint(f, y(f))
    }) to coefficients
}

