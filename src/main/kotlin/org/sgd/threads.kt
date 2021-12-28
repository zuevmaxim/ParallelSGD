package org.sgd

import net.openhft.affinity.Affinity
import oshi.SystemInfo

val numaConfig: Map<Int, List<Int>> = SystemInfo().hardware.processor.logicalProcessors
    .groupBy { it.numaNode }
    .mapValues { cpus -> cpus.value.map { cpu -> cpu.processorNumber } }

val threadToNumaNode = numaConfig.entries.flatMap { node -> node.value.map { it to node.key } }.toMap()


fun bindCurrentThreadToCpu(cpuId: Int) {
    Affinity.setAffinity(cpuId)
}
