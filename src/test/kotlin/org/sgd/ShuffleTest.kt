package org.sgd

import org.junit.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals


internal class ShuffleTest {

    @Test
    fun shuffle() {
        val list = (1..10).toList()
        val c = list.toMutableList()
        repeat(100) {
            c.shuffle()
            assertEquals(list.size, c.size)
            for (e in list) {
                assertContains(c, e)
            }
        }
    }
}
