package org.sgd

import java.io.File
import kotlin.random.Random
import kotlin.system.measureTimeMillis

typealias Type = Float
typealias TypeArray = FloatArray
fun Collection<Type>.toArray() = toFloatArray()
fun String.toWorkingType(): Type = toFloat()
const val ONE: Type = 1f
const val ZERO: Type = 0f
fun Double.toType(): Type = toFloat()
fun Int.toType(): Type = toFloat()

class DataPoint(val indices: IntArray, val xValues: TypeArray, val y: Type)

class DataSet(val points: List<DataPoint>) {
    val size get() = points.size

    fun split(part: Double): Pair<DataSet, DataSet> {
        require(0.0 < part && part < 1.0)
        val n = (size * part).toInt()
        return DataSet(points.subList(0, n).toList()) to DataSet(points.subList(n, size).toList())
    }
}

interface DataPointLoss {
    fun loss(w: TypeArray): Type
    fun gradientStep(w: TypeArray, learningRate: Type)
}

abstract class DataSetLoss(dataSet: DataSet) {
    val pointLoss = dataSet.points.map { pointLoss(it) }

    protected abstract fun pointLoss(p: DataPoint): DataPointLoss

    fun loss(w: TypeArray): Type {
        var s: Type = ZERO
        val points = pointLoss
        repeat(points.size) {
            s += points[it].loss(w)
        }
        return s / points.size
    }
}


fun loadDataSet(file: File): DataSet {
    val points = mutableListOf<DataPoint>()
    val timeMs = measureTimeMillis {
        file.useLines { lines ->
            lines.forEach { line ->
                val parts = line.trim().split(" ")
                val y = if (parts[0].toInt() == 1) ONE else ZERO
                val (indices, values) = parts.drop(1).map { val (id, count) = it.split(':'); id.toInt() to count.toWorkingType() }.unzip()
                points.add(DataPoint(indices.toIntArray(), values.toArray(), y))
            }
        }
    }
    println("Dataset ${file.name} loaded in ${timeMs / 1000} s")
    return DataSet(points)
}

fun TypeArray.resetToZero() {
    repeat(size) {
        this[it] = ZERO
    }
}

fun TypeArray.subtract(x: TypeArray): TypeArray {
    require(size == x.size)
    repeat(size) { i ->
        this[i] -= x[i]
    }
    return this
}

fun TypeArray.add(x: TypeArray): TypeArray {
    require(size == x.size)
    repeat(size) { i ->
        this[i] += x[i]
    }
    return this
}

fun TypeArray.multiply(x: Type): TypeArray {
    repeat(size) { i ->
        this[i] *= x
    }
    return this
}

fun TypeArray.divide(x: Type) = multiply(1 / x)

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
