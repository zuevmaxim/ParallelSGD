package org.sgd

import kotlin.math.exp
import kotlin.math.pow

class LinearRegressionLoss(private val dataSet: DataSet) : DataSetLoss(dataSet) {
    override fun pointLoss(p: DataPoint) = LinearRegressionPointLoss(p)

    fun precision(w: TypeArray): Pair<Double, Double> {
        var truePositive = 0
        var falsePositive = 0
        var trueNegative = 0
        var falseNegative = 0
        for (point in dataSet.points) {
            val prediction = if (dot(w, point.indices, point.xValues) >= ZERO) ONE else ZERO
            if (prediction == ONE) {
                if (prediction == point.y) truePositive++ else falsePositive++
            } else {
                if (prediction == point.y) trueNegative++ else falseNegative++
            }
        }
        return truePositive.toDouble() / (truePositive + falseNegative) to trueNegative.toDouble() / (trueNegative + falsePositive)
    }
}

class LinearRegressionPointLoss(private val p: DataPoint) : DataPointLoss {
    override fun loss(w: TypeArray): Type = (p.y - if (dot(w, p.indices, p.xValues) >= ZERO) ONE else ZERO).pow(2)

    override fun gradientStep(w: TypeArray, learningRate: Type) {
        val indices = p.indices
        val xs = p.xValues
        val e = exp(-dot(w, indices, xs))
        val grad = learningRate * (1 / (1 + e) - p.y)
        repeat(indices.size) { i ->
            w[indices[i]] -= xs[i] * grad
        }
        w[w.size - 1] -= grad
    }
}

private fun dot(w: TypeArray, indices: IntArray, xs: TypeArray): Type {
    var s = w[w.size - 1]
    repeat(indices.size) { i ->
        s += xs[i] * w[indices[i]]
    }
    return s
}
