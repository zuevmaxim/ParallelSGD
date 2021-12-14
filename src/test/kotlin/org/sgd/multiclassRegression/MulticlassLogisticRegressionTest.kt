package org.sgd.multiclassRegression

import org.sgd.loadBinaryDataSet
import org.sgd.logisticRegression.baseDir
import java.io.File

private const val DATASET = "rcv1"


private val dataset by lazy { loadBinaryDataSet(File(baseDir, DATASET), File(baseDir, "$DATASET.t")) }
private val features by lazy { dataset.first.points.asSequence().plus(dataset.second.points).maxOf { it.indices.maxOrNull() ?: 0 } }

