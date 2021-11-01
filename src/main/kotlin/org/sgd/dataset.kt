package org.sgd

import kotlin.random.Random

typealias Features = DoubleArray
typealias Value = Double
typealias LossValue = Double
typealias Weights = DoubleArray

class DataPoint(val x: Features, val y: Value) {
    val dimension get() = x.size
}

class DataSet(val points: Array<DataPoint>) {
    init {
        require(points.all { it.dimension == points[0].dimension }) { "DataSet points must have same dimension." }
    }

    val size get() = points.size

    fun split(part: Double): Pair<DataSet, DataSet> {
        require(0.0 < part && part < 1.0)
        val n = (size * part).toInt()
        return DataSet(points.copyOfRange(0, n)) to DataSet(points.copyOfRange(n, size))
    }
}

interface DataPointLoss {
    fun loss(w: Weights): LossValue
    fun grad(w: Weights, buffer: Weights): Weights
}

abstract class DataSetLoss(dataSet: DataSet) {
    val pointLoss = dataSet.points.map { pointLoss(it) }

    protected abstract fun pointLoss(p: DataPoint): DataPointLoss

    fun loss(w: Weights): LossValue = pointLoss.sumOf { it.loss(w) } / pointLoss.size
}


fun DoubleArray.resetToZero() {
    repeat(size) {
        this[it] = 0.0
    }
}

fun DoubleArray.subtract(x: DoubleArray): DoubleArray {
    require(size == x.size)
    repeat(size) { i ->
        this[i] -= x[i]
    }
    return this
}

fun DoubleArray.add(x: DoubleArray): DoubleArray {
    require(size == x.size)
    repeat(size) { i ->
        this[i] += x[i]
    }
    return this
}

fun DoubleArray.multiply(x: Double): DoubleArray {
    repeat(size) { i ->
        this[i] *= x
    }
    return this
}

fun DoubleArray.divide(x: Double) = multiply(1 / x)

fun DoubleArray.dot(x: DoubleArray): Double {
    require(size == x.size)
    var result = 0.0
    repeat(size) { i ->
        result += this[i] * x[i]
    }
    return result
}

inline fun repeat(n: Int, action: (Int) -> Unit) {
    var i = 0
    while (i < n) {
        action(i)
        i++
    }
}

fun <T> MutableList<T>.shuffle() {
    repeat(size) { i ->
        val j = Random.nextInt(i, size)
        this[i] = this[j]
            .also { this[j] = this[i] }
    }
}
