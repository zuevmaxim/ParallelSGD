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

val params: Map<String, Map<String, Type>> = hashMapOf(
    "w8a" to hashMapOf(
        "learningRate" to 0.26.toType(),
        "stepDecay" to 1.0.toType(),
        "targetLoss" to 0.02.toType(),
        "batch" to 10.toType(),
    ),
    "rcv1" to hashMapOf(
        "learningRate" to 0.5.toType(),
        "stepDecay" to 0.8.toType(),
        "targetLoss" to 0.0236.toType(),
        "batch" to 20.toType(),
    ),
    "webspam" to hashMapOf(
        "learningRate" to 0.2.toType(),
        "stepDecay" to 0.8.toType(),
        "targetLoss" to 0.0756.toType(),
        "batch" to 10.toType(),
    ),
    "news20" to hashMapOf(
        "learningRate" to 0.5.toType(),
        "stepDecay" to 0.8.toType(),
        "targetLoss" to 0.1632.toType(),
        "batch" to 10.toType(),
    ),
    "covtype" to hashMapOf(
        "learningRate" to 0.005.toType(),
        "stepDecay" to 0.85.toType(),
        "targetLoss" to 0.2846.toType(),
        "batch" to 10.toType(),
    ),
)

val models: Map<String, () -> Pair<Model, Model>> = hashMapOf(
    "w8a" to ::createBinaryModel,
    "rcv1" to ::createBinaryModel,
    "webspam" to ::createBinaryModel,
    "news20" to ::createMulticlassModel,
    "covtype" to ::createMulticlassModel
).mapValues { { it.value(it.key) } }

fun logSequence(maxValue: Int, step: Double = 2.0): List<Int> {
    var value = maxValue.toDouble()
    val result = hashSetOf<Int>()
    while (value >= 1.0) {
        result.add(value.roundToInt())
        value /= step
    }
    return result.toList().sorted()
}

private val datasets = hashMapOf<String, Any>()

private fun createBinaryModel(datasetName: String): Pair<Model, Model> {
    val dataset = datasets.getOrPut(datasetName) { loadBinaryDataSet(File(baseDir, datasetName), File(baseDir, "$datasetName.t")) } as Pair<DataSet, DataSet>
    val features = features(dataset.first, dataset.second)
    val trainLoss = LogisticRegressionModel(dataset.first, features)
    val testLoss = LogisticRegressionModel(dataset.second, features)
    return trainLoss to testLoss
}

private fun createMulticlassModel(datasetName: String): Pair<Model, Model> {
    val dataset = datasets.getOrPut(datasetName) { loadMulticlassDataSet(File(baseDir, datasetName), File(baseDir, "$datasetName.t")) } as Triple<DataSet, DataSet, Int>
    val features = features(dataset.first, dataset.second)
    val trainLoss = MulticlassLogisticRegressionModel(dataset.first, dataset.third, features)
    val testLoss = MulticlassLogisticRegressionModel(dataset.second, dataset.third, features)
    return trainLoss to testLoss
}

private fun features(train: DataSet, test: DataSet) = train.points.asSequence().plus(test.points).maxOf { it.indices.maxOrNull() ?: 0 }

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
