package me.kright.lists

import scala.annotation.unchecked.uncheckedVariance

/**
 * Hand-written immutable single-linked list, one element per node.
 * Analog of scala.List, used as a baseline for chunked lists.
 */
sealed trait MyList[+A]:
  def isEmpty: Boolean

  def ::[B >: A](elem: B): MyList[B] = new MyCons(elem, this)

  def foreach(f: A => Unit): Unit

  def ++[B >: A](that: MyList[B]): MyList[B]

  def length: Int =
    var result = 0
    var current: MyList[A] = this
    while current.isInstanceOf[MyCons[?]] do
      result += 1
      current = current.asInstanceOf[MyCons[A]].tail
    result

object MyList:
  def empty[A]: MyList[A] = MyNil

  def apply[A](elems: A*): MyList[A] =
    var result: MyList[A] = MyNil
    var i = elems.length - 1
    while i >= 0 do
      result = elems(i) :: result
      i -= 1
    result

case object MyNil extends MyList[Nothing]:
  override def isEmpty: Boolean = true

  override def foreach(f: Nothing => Unit): Unit = ()

  override def ++[B](that: MyList[B]): MyList[B] = that

// tailVar is mutable for the same reason as in scala.List:
// ++ builds copied nodes iteratively and links them in place, without recursion.
final class MyCons[+A](val head: A, private[lists] var tailVar: MyList[A @uncheckedVariance]) extends MyList[A]:
  def tail: MyList[A] = tailVar

  override def isEmpty: Boolean = false

  override def foreach(f: A => Unit): Unit =
    var current: MyList[A] = this
    while current.isInstanceOf[MyCons[?]] do
      val c = current.asInstanceOf[MyCons[A]]
      f(c.head)
      current = c.tail

  override def ++[B >: A](that: MyList[B]): MyList[B] =
    if that.isEmpty then this
    else
      val first = new MyCons[B](head, that)
      var last = first
      var current = tail
      while current.isInstanceOf[MyCons[?]] do
        val c = current.asInstanceOf[MyCons[A]]
        val copied = new MyCons[B](c.head, that)
        last.tailVar = copied
        last = copied
        current = c.tail
      first

  // name-based extractor support: `case MyCons(head, tail)` without Option/Tuple allocation
  def get: MyCons[A] = this
  def _1: A = head
  def _2: MyList[A] = tail

object MyCons:
  def unapply[A](cons: MyCons[A]): MyCons[A] = cons
