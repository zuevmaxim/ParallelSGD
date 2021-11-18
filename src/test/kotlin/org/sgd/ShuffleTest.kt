package org.sgd

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test


internal class ShuffleTest {

    @Test
    fun shuffle() {
        val list = (1..10).toList()
        val c = list.toMutableList()
        repeat(100) {
            c.shuffle()
            assertEquals(list.size, c.size)
            for (e in list) {
                assertTrue(c.contains(e))
            }
        }
    }
}
