package org.sgd

import kotlin.math.abs
import kotlin.math.exp

class LinearRegressionLoss : Loss {
    override fun pointLoss(p: DataPoint) = LinearRegressionPointLoss(p)
}

class LinearRegressionPointLoss(private val p: DataPoint) : DataPointLoss {
    override fun loss(w: TypeArray): Type = abs(p.y - predict(w, p.indices, p.xValues))

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

    override fun predict(w: TypeArray, indices: IntArray, xValues: TypeArray) =
        if (dot(w, indices, xValues) >= ZERO) ONE else ZERO
}

private fun dot(w: TypeArray, indices: IntArray, xs: TypeArray): Type {
    var s = w[w.size - 1]
    repeat(indices.size) { i ->
        s += xs[i] * w[indices[i]]
    }
    return s
}
