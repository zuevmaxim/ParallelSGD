package org.sgd

import kotlin.math.abs
import kotlin.math.exp


class LogisticRegressionModel(dataSet: DataSet, private val features: Int) : Model(dataSet) {
    override fun loss(p: DataPoint, w: TypeArray): Type = abs(p.y - predict(w, p.indices, p.xValues))

    override fun gradientStep(p: DataPoint, w: TypeArray, learningRate: Type) {
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

    override fun createWeights() = TypeArray(features + 1)
}

class MulticlassLogisticRegressionModel(dataSet: DataSet, private val numberOfClasses: Int, features: Int) : Model(dataSet) {
    private val features = features + 1

    override fun loss(p: DataPoint, w: TypeArray): Type = if (p.y == predict(w, p.indices, p.xValues)) ZERO else ONE
    private val buffer = ThreadLocal.withInitial { TypeArray(numberOfClasses - 1) }

    override fun gradientStep(p: DataPoint, w: TypeArray, learningRate: Type) {
        val indices = p.indices
        val xs = p.xValues
        val n = features

        val es = calc(n, w, indices, xs)
        val k = p.y.toInt()
        if (k < es.size) {
            es[k] -= ONE
        }

        repeat(es.size) { i ->
            es[i] *= learningRate
            val offset = n * i
            repeat(indices.size) { j ->
                w[offset + indices[j]] -= xs[j] * es[i]
            }
            w[offset + n - 1] -= es[i]
        }
    }

    private fun calc(n: Int, w: TypeArray, indices: IntArray, xs: TypeArray): TypeArray {
        val es = buffer.get()
        val cls = es.size
        var sum = ZERO
        repeat(cls) { i ->
            val e = exp(dot(i, n, w, indices, xs))
            es[i] = e
            sum += e
        }
        sum += ONE
        repeat(cls) { i ->
            es[i] = es[i] / sum
        }
        return es
    }

    override fun predict(w: TypeArray, indices: IntArray, xValues: TypeArray): Type {
        val probabilities = calc(features, w, indices, xValues)
        var lastProbability = ONE
        var maxProb = ZERO
        var maxIndex = 0
        repeat(probabilities.size) { i ->
            val p = probabilities[i]
            lastProbability -= p
            if (p > maxProb) {
                maxIndex = i
                maxProb = p
            }
        }
        if (lastProbability > maxProb) {
            maxIndex = probabilities.size
            maxProb = lastProbability
        }
        return maxIndex.toType()
    }

    override fun createWeights() = TypeArray((numberOfClasses - 1) * features)
}

private fun dot(w: TypeArray, indices: IntArray, xs: TypeArray): Type {
    var s = ZERO
    repeat(indices.size) { i ->
        s += xs[i] * w[indices[i]]
    }
    s += w[w.size - 1]
    return s
}

private fun dot(classIndex: Int, features: Int, w: TypeArray, indices: IntArray, xs: TypeArray): Type {
    val offset = classIndex * features
    var s = ZERO
    repeat(indices.size) { i ->
        s += xs[i] * w[offset + indices[i]]
    }
    s += w[offset + features - 1]
    return s
}
