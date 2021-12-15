package org.sgd

import java.io.File
import kotlin.random.Random
import kotlin.system.measureTimeMillis

typealias Type = Float
typealias TypeArray = FloatArray

fun String.toType(): Type = toFloat()
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


abstract class Model(dataSet: DataSet) {
    val points = dataSet.points
    abstract fun loss(p: DataPoint, w: TypeArray): Type
    abstract fun gradientStep(p: DataPoint, w: TypeArray, learningRate: Type)
    abstract fun predict(w: TypeArray, indices: IntArray, xValues: TypeArray): Type
    abstract fun createWeights(): TypeArray

    fun loss(w: TypeArray): Type {
        var s: Type = ZERO
        val points = points
        repeat(points.size) {
            s += loss(points[it], w)
        }
        return s / points.size
    }

    fun precision(w: TypeArray): Pair<Double, Double> {
        var truePositive = 0
        var falsePositive = 0
        var trueNegative = 0
        var falseNegative = 0
        for (point in points) {
            val prediction = predict(w, point.indices, point.xValues)
            if (prediction == ONE) {
                if (prediction == point.y) truePositive++ else falsePositive++
            } else {
                if (prediction == point.y) trueNegative++ else falseNegative++
            }
        }
        return truePositive.toDouble() / (truePositive + falseNegative) to trueNegative.toDouble() / (trueNegative + falsePositive)
    }
}

private inline fun loadDataSet(file: File, preprocessLabels: (String) -> Type): DataSet {
    val points = mutableListOf<DataPoint>()
    val timeMs = measureTimeMillis {
        file.useLines { lines ->
            lines.forEach { line ->
                val parts = line.trim().split(" ")
                val indices = IntArray(parts.size - 1)
                val values = TypeArray(parts.size - 1)
                repeat(parts.size - 1) { index ->
                    val (id, value) = parts[index + 1].split(':')
                    indices[index] = id.toInt()
                    values[index] = value.toType()
                }
                points.add(DataPoint(indices, values, preprocessLabels(parts[0])))
            }
        }
    }
    println("Dataset ${file.name} loaded in ${timeMs / 1000} s")
    return DataSet(points)
}

private fun loadBinaryDataSet(file: File) = loadDataSet(file) { label ->
    if (label.toInt() == 1) ONE else ZERO
}

fun loadBinaryDataSet(train: File, test: File) =  loadBinaryDataSet(train) to loadBinaryDataSet(test)

fun loadMulticlassDataSet(train: File, test: File): Triple<DataSet, DataSet, Int> {
    val keyMap = hashMapOf<Int, Type>()
    return Triple(loadDataSet(train) {
        keyMap.getOrPut(it.toInt()) { keyMap.size.toType() }
    }, loadDataSet(test) {
        keyMap[it.toInt()] ?: error("There is no label $it in train dataset!")
    }, keyMap.size)
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
