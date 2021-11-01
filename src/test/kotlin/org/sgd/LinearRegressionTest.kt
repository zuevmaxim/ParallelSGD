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
    fun solverCompare() {
        val timeS = mutableListOf<Double>()
        val trainLoss = mutableListOf<Double>()
        val solverNames = mutableListOf<String>()

        val iterationsNumber = 30
        val solvers = listOf(
            SimpleSGDSolver(iterationsNumber, 0.0002),
            SimpleParallelSGDSolver(iterationsNumber, 0.00004, 12),
            ClusterParallelSGDSolver(iterationsNumber, 0.00004, 12, 4)
        )

        val features = 10000
        val size = 40000
        val (dataset, coefficients) = generateRandomDataSet(features, size)
        val (test, train) = dataset.split(0.5)
        val loss = LinearRegressionLoss(train)

        println("Dataset loss: " + LinearRegressionLoss(test).loss(coefficients))
        for (solver in solvers) {
            val name = solver.javaClass.simpleName
            val result = solver.solve(loss, DoubleArray(features + 1))

            println("$name Test loss: " + LinearRegressionLoss(test).loss(result.w))

            for ((timeNs, lossValue) in result.timeNsToLoss.filterValues { it < 10.0 }) {
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

