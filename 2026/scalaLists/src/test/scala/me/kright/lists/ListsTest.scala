package me.kright.lists

import org.junit.Assert.*
import org.junit.Test

import scala.collection.mutable.ArrayBuffer

class ListsTest {

  private val sizes = 0 to 20

  private def foreachToSeq[A](foreachF: (A => Unit) => Unit): Seq[A] = {
    val buf = ArrayBuffer[A]()
    foreachF(buf += _)
    buf.toSeq
  }

  private def unapplyToSeqMy[A](list: MyList[A]): Seq[A] = {
    val buf = ArrayBuffer[A]()
    var current = list
    var continue = true
    while (continue) {
      current match {
        case MyCons(h, t) =>
          buf += h
          current = t
        case _ =>
          continue = false
      }
    }
    buf.toSeq
  }

  private def unapplyToSeq2[A](list: List2[A]): Seq[A] = {
    val buf = ArrayBuffer[A]()
    var current = list
    var continue = true
    while (continue) {
      current match {
        case Node2(h, t) =>
          buf += h
          current = t
        case _ =>
          continue = false
      }
    }
    buf.toSeq
  }

  private def unapplyToSeq4[A](list: List4[A]): Seq[A] = {
    val buf = ArrayBuffer[A]()
    var current = list
    var continue = true
    while (continue) {
      current match {
        case Node4(h, t) =>
          buf += h
          current = t
        case _ =>
          continue = false
      }
    }
    buf.toSeq
  }

  private def unapplyToSeq8[A](list: List8[A]): Seq[A] = {
    val buf = ArrayBuffer[A]()
    var current = list
    var continue = true
    while (continue) {
      current match {
        case Node8(h, t) =>
          buf += h
          current = t
        case _ =>
          continue = false
      }
    }
    buf.toSeq
  }

  @Test
  def myListForeachAndUnapply(): Unit =
    for (n <- sizes) {
      val expected = 0 until n
      val list = MyList(expected*)
      assertEquals(expected, foreachToSeq[Int](list.foreach))
      assertEquals(expected, unapplyToSeqMy(list))
      assertEquals(n, list.length)
    }

  @Test
  def list2ForeachAndUnapply(): Unit =
    for (n <- sizes) {
      val expected = 0 until n
      val list = List2(expected*)
      assertEquals(expected, foreachToSeq[Int](list.foreach))
      assertEquals(expected, unapplyToSeq2(list))
      assertEquals(n, list.length)
    }

  @Test
  def list4ForeachAndUnapply(): Unit =
    for (n <- sizes) {
      val expected = 0 until n
      val list = List4(expected*)
      assertEquals(expected, foreachToSeq[Int](list.foreach))
      assertEquals(expected, unapplyToSeq4(list))
      assertEquals(n, list.length)
    }

  @Test
  def list8ForeachAndUnapply(): Unit =
    for (n <- sizes) {
      val expected = 0 until n
      val list = List8(expected*)
      assertEquals(expected, foreachToSeq[Int](list.foreach))
      assertEquals(expected, unapplyToSeq8(list))
      assertEquals(n, list.length)
    }

  @Test
  def myListConcat(): Unit =
    for (n <- sizes; m <- sizes) {
      val left = 0 until n
      val right = n until (n + m)
      val result = MyList(left*) ++ MyList(right*)
      assertEquals(left ++ right, foreachToSeq[Int](result.foreach))
      assertEquals(left ++ right, unapplyToSeqMy(result))
    }

  @Test
  def list2Concat(): Unit =
    for (n <- sizes; m <- sizes) {
      val left = 0 until n
      val right = n until (n + m)
      val result = List2(left*) ++ List2(right*)
      assertEquals(left ++ right, foreachToSeq[Int](result.foreach))
      assertEquals(left ++ right, unapplyToSeq2(result))
    }

  @Test
  def list4Concat(): Unit =
    for (n <- sizes; m <- sizes) {
      val left = 0 until n
      val right = n until (n + m)
      val result = List4(left*) ++ List4(right*)
      assertEquals(left ++ right, foreachToSeq[Int](result.foreach))
      assertEquals(left ++ right, unapplyToSeq4(result))
    }

  @Test
  def list8Concat(): Unit =
    for (n <- sizes; m <- sizes) {
      val left = 0 until n
      val right = n until (n + m)
      val result = List8(left*) ++ List8(right*)
      assertEquals(left ++ right, foreachToSeq[Int](result.foreach))
      assertEquals(left ++ right, unapplyToSeq8(result))
      assertEquals(n + m, result.length)
    }

  @Test
  def concatDoesNotModifyArguments(): Unit = {
    val left = List8(1, 2, 3, 4, 5)
    val right = List8(6, 7, 8)
    left ++ right
    assertEquals(Seq(1, 2, 3, 4, 5), foreachToSeq[Int](left.foreach))
    assertEquals(Seq(6, 7, 8), foreachToSeq[Int](right.foreach))
  }

  @Test
  def prependAfterTail(): Unit = {
    // tail creates a shifted node; prepending to it must not lose elements
    for (n <- 1 to 20) {
      val expected = 0 until n
      List4(expected*) match {
        case Node4(head, tail) =>
          assertEquals(0, head)
          val rebuilt = -1 :: tail
          assertEquals(-1 +: (expected.tail), foreachToSeq[Int](rebuilt.foreach))
        case _ =>
          fail(s"list of size $n should not be empty")
      }
    }
  }

  @Test
  def chunkedNodesAreFilled(): Unit = {
    // when built by prepending, all nodes except the head one must be full
    var list: List8[Int] = List8((0 until 100)*)
    list match {
      case node: Node8[Int] =>
        var current: List8[Int] = node.next
        while (current.isInstanceOf[Node8[?]]) {
          val n = current.asInstanceOf[Node8[Int]]
          assertEquals(8, n.size)
          current = n.next
        }
      case _ => fail("should not be empty")
    }
  }
}
