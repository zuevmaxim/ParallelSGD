package org.sgd

import kotlinx.atomicfu.AtomicIntArray
import java.io.File
import kotlin.math.min
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

class DataSet(val points: Array<DataPoint>) {
    val size get() = points.size

    fun split(part: Double): Pair<DataSet, DataSet> {
        require(0.0 < part && part < 1.0)
        val n = (size * part).toInt()
        return DataSet(points.copyOfRange(0, n)) to DataSet(points.copyOfRange(n, size))
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
    return DataSet(points.toTypedArray())
}

private fun loadBinaryDataSet(file: File) = loadDataSet(file) { label ->
    if (label.toInt() == 1) ONE else ZERO
}

fun loadBinaryDataSet(train: File, test: File) = loadBinaryDataSet(train) to loadBinaryDataSet(test)

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

inline fun iterate(start: Int, end: Int, action: (Int) -> Unit) {
    var i = start
    while (i < end) {
        action(i)
        i++
    }
}

fun <T> Array<T>.shuffle(start: Int = 0, end: Int = size) {
    val n = end - start
    repeat(n) { i ->
        val j = Random.nextInt(i, n)
        swap(start + i, start + j)
    }
}

fun <T> Array<T>.swap(i: Int, j: Int) {
    this[i] = this[j]
        .also { this[j] = this[i] }
}

fun <T> parallelShuffleTask(points: Array<T>, threadId: Int, threads: Int, status: AtomicIntArray) {
    val n = points.size
    if (n < 1000 || threads == 1) {
        points.shuffle()
        return
    }
    val block = n / threads
    val start = block * threadId
    val end = if (threadId == threads - 1) n else block * (threadId + 1)

    status[threadId].value = 0
    points.shuffle(start, end)
    var level = 1
    while (true) {
        status[threadId].value = level
        if ((threadId ushr (level - 1) and 1) != 0) break
        val blockCount = 1 shl (level - 1)
        val neighbour = threadId + blockCount
        if (neighbour < threads) {
            while (status[neighbour].value != level);
            val right = start + blockCount * block
            val rightEnd = min(right + blockCount * block, n)
            points.mergeShuffle(start, right, rightEnd)
        } else if (threadId == 0) break
        level++
    }
}

private fun <T> Array<T>.mergeShuffle(left: Int, right: Int, end: Int) {
    var i = left
    var j = right
    while (true) {
        if (Random.nextBoolean()) {
            if (i == j) break
        } else {
            if (j == end) break
            swap(i, j)
            j++
        }
        i++
    }
    while (i < end) {
        val m = Random.nextInt(left, i)
        swap(i, m)
        i++
    }
}
