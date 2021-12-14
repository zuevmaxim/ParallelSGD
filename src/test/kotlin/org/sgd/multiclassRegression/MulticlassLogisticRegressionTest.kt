package org.sgd.multiclassRegression

import kotlinx.smartbench.benchmark.Benchmark
import kotlinx.smartbench.declarative.Operation
import org.sgd.*
import org.sgd.logisticRegression.baseDir
import java.io.File

private const val AVERAGE_LOSS_METRIC = "average loss"
private const val SPEEDUP_METRIC = "speedup"
private const val CLUSTER_METHOD_PREFIX = "cluster-"

private const val DATASET = "rcv1"


private val dataset by lazy { loadBinaryDataSet(File(baseDir, DATASET), File(baseDir, "$DATASET.t")) }
private val features by lazy { dataset.first.points.asSequence().plus(dataset.second.points).maxOf { it.indices.maxOrNull() ?: 0 } }

class RunRegressionTask(
    val method: String, val learningRate: Type, val stepDecay: Type, val workingThreads: Int, val targetLoss: Type
) : Benchmark() {
    private val trainLoss = DataSetLoss(dataset.first, LogisticRegressionLoss())
    private val testLoss = DataSetLoss(dataset.second, LogisticRegressionLoss())

    @Operation
    fun run(): Type {
        val solver = when {
            method == "simple" -> ParallelSGDSolver(learningRate, workingThreads, stepDecay)
            method.startsWith(CLUSTER_METHOD_PREFIX) -> ClusterParallelSGDSolver(learningRate, workingThreads, stepDecay, method.substring(CLUSTER_METHOD_PREFIX.length).toInt())
            else -> error("Unknown method $method")
        }
        val result = solver.solve(trainLoss, testLoss, TypeArray(features + 1), targetLoss)
        return testLoss.loss(result.w)
    }
}

class SequentialRegressionTask(
    val learningRate: Type, val stepDecay: Type, val iterations: Int
) : Benchmark() {
    private val trainLoss = DataSetLoss(dataset.first, LogisticRegressionLoss())
    private val testLoss = DataSetLoss(dataset.second, LogisticRegressionLoss())
    var loss = BenchmarkCounter()
    val counter = BenchmarkCounter()

    @Operation
    fun run(): Type {
        val solver = SequentialSGDSolver(iterations, learningRate, stepDecay)
        val result = solver.solve(trainLoss, testLoss, TypeArray(features + 1), ZERO)
        return testLoss.loss(result.w).also {
            val scale = 1e3.toLong()
            loss.inc((it * scale).toLong())
            counter.inc(scale)
        }
    }
}

class MulticlassLogisticRegressionTest {

}
