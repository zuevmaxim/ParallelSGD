package org.sgd

import java.io.File
import kotlin.random.Random
import kotlin.system.measureTimeMillis

typealias LossValue = Double
typealias Weights = DoubleArray

class DataPoint(val indices: IntArray, val xValues: DoubleArray, val y: Double)

class DataSet(val points: List<DataPoint>) {
    val size get() = points.size

    fun split(part: Double): Pair<DataSet, DataSet> {
        require(0.0 < part && part < 1.0)
        val n = (size * part).toInt()
        return DataSet(points.subList(0, n).toList()) to DataSet(points.subList(n, size).toList())
    }
}

interface DataPointLoss {
    fun loss(w: Weights): LossValue
    fun gradientStep(w: Weights, learningRate: Double)
}

abstract class DataSetLoss(dataSet: DataSet) {
    val pointLoss = dataSet.points.map { pointLoss(it) }

    protected abstract fun pointLoss(p: DataPoint): DataPointLoss

    fun loss(w: Weights): LossValue = pointLoss.sumOf { it.loss(w) } / pointLoss.size
}


fun loadDataSet(file: File): DataSet {
    val points = mutableListOf<DataPoint>()
    val timeMs = measureTimeMillis {
        file.useLines { lines ->
            lines.forEach { line ->
                val parts = line.trim().split(" ")
                val y = if (parts[0].toInt() == 1) 1.0 else 0.0
                val (indices, values) = parts.drop(1).map { val (id, count) = it.split(':'); id.toInt() to count.toDouble() }.unzip()
                points.add(DataPoint(indices.toIntArray(), values.toDoubleArray(), y))
            }
        }
    }
    println("Dataset ${file.name} loaded in $timeMs ms")
    return DataSet(points)
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
