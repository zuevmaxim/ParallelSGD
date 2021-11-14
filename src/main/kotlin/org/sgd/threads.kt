package org.sgd

import net.openhft.affinity.Affinity
import oshi.SystemInfo

val numaConfig: Map<Int, List<Int>> = SystemInfo().hardware.processor.logicalProcessors
    .groupBy { it.numaNode }
    .mapValues { cpus -> cpus.value.map { cpu -> cpu.processorNumber } }


fun bindCurrentThreadToCpu(cpuId: Int) {
    Affinity.setAffinity(cpuId)
}
