package org.sgd

import java.io.File
import kotlin.math.roundToInt

const val AVERAGE_LOSS_METRIC = "average loss"
const val SPEEDUP_METRIC = "speedup"
const val CLUSTER_METHOD_PREFIX = "cluster-"

val baseDir = File("../datasets").let {
    if (it.exists()) it
    else File("/home/maksim.zuev/datasets")
}

val params = hashMapOf(
    "rcv1" to hashMapOf(
        "learningRate" to 0.5.toType(),
        "stepDecay" to 0.8.toType(),
        "targetLoss" to 0.024.toType()
    ),
    "webspam" to hashMapOf(
        "learningRate" to 0.2.toType(),
        "stepDecay" to 0.8.toType(),
        "targetLoss" to 0.0756.toType()
    ),
    "news20" to hashMapOf(
        "learningRate" to 0.5.toType(),
        "stepDecay" to 0.8.toType(),
        "targetLoss" to 0.1632.toType()
    ),
    "covtype" to hashMapOf(
        "learningRate" to 0.005.toType(),
        "stepDecay" to 0.85.toType(),
        "targetLoss" to 0.2846.toType()
    ),
)

fun logSequence(maxValue: Int, step: Double = 2.0): List<Int> {
    var value = maxValue.toDouble()
    val result = hashSetOf<Int>()
    while (value >= 1.0) {
        result.add(value.roundToInt())
        value /= step
    }
    return result.toList().sorted()
}

fun splitDataset(name: String) {
    val source = "../datasets/$name"
    val train = "../datasets/$name"
    val test = "../datasets/$name.t"
    File(source).useLines {
        val strings = it.toMutableList()
        strings.shuffle()
        File(train).run {
            createNewFile()
            writeText(strings.subList(0, (strings.size * 0.8).toInt()).joinToString("\n"))
        }
        File(test).run {
            createNewFile()
            writeText(strings.subList((strings.size * 0.8).toInt(), strings.size).joinToString("\n"))
        }
    }
}
