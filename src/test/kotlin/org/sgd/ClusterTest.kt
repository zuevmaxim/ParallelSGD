package org.sgd

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import kotlin.math.min

class ClusterTest {
    @Test
    fun testDivision() {
        val threadToNuma = numaConfig.flatMap { (nodeId, cpus) -> cpus.map { it to nodeId } }.toMap()

        for (threadsPerCluster in logSequence(numaConfig.values.maxOf { it.size })) {
            for (threads in logSequence(Runtime.getRuntime().availableProcessors())) {
                val clusters = extractClusters(threads, threadsPerCluster)
                Assertions.assertEquals(threads, clusters.sumOf { it.size })
                Assertions.assertEquals(threads, clusters.flatten().toSet().size)

                for (cluster in clusters) {
                    Assertions.assertEquals(min(threadsPerCluster, threads), cluster.size)
                    val numaId = threadToNuma[cluster[0]]!!
                    for (threadId in cluster) {
                        Assertions.assertEquals(numaId, threadToNuma[threadId])
                    }
                }
            }
        }
    }
}
