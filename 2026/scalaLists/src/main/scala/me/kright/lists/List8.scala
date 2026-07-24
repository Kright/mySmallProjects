package me.kright.lists

import scala.annotation.unchecked.uncheckedVariance

/**
 * Immutable single-linked list where each node stores up to 8 elements.
 * Elements are kept in fields (not in an array) to avoid an extra indirection.
 * Only e0 is guaranteed to be present, e1..e7 are typed as `A | Null`:
 * slots at indices >= size hold nulls.
 */
sealed trait List8[+A]:
  def isEmpty: Boolean

  def ::[B >: A](elem: B): List8[B]

  def foreach(f: A => Unit): Unit

  def ++[B >: A](that: List8[B]): List8[B]

  /** sums up the sizes of the nodes without touching the elements */
  def length: Int =
    var result = 0
    var current: List8[A] = this
    while current.isInstanceOf[Node8[?]] do
      val node = current.asInstanceOf[Node8[A]]
      result += node.size
      current = node.next
    result

object List8:
  def empty[A]: List8[A] = Nil8

  def apply[A](elems: A*): List8[A] =
    var result: List8[A] = Nil8
    var i = elems.length - 1
    while i >= 0 do
      result = elems(i) :: result
      i -= 1
    result

case object Nil8 extends List8[Nothing]:
  override def isEmpty: Boolean = true

  override def ::[B](elem: B): List8[B] =
    new Node8(1, elem, null, null, null, null, null, null, null, Nil8)

  override def foreach(f: Nothing => Unit): Unit = ()

  override def ++[B](that: List8[B]): List8[B] = that

// nextVar is mutable for the same reason as in scala.List:
// ++ builds copied nodes iteratively and links them in place, without recursion.
final class Node8[+A](val size: Int,
                      val e0: A,
                      val e1: A | Null, val e2: A | Null, val e3: A | Null,
                      val e4: A | Null, val e5: A | Null, val e6: A | Null, val e7: A | Null,
                      private[lists] var nextVar: List8[A @uncheckedVariance]) extends List8[A]:
  def next: List8[A] = nextVar

  override def isEmpty: Boolean = false

  def head: A = e0

  /** the tail node is created dynamically: same elements shifted by one, without the head */
  def tail: List8[A] =
    if size == 1 then next
    else new Node8(size - 1, e1.asInstanceOf[A], e2, e3, e4, e5, e6, e7, null, next)

  override def ::[B >: A](elem: B): List8[B] =
    if size < 8 then new Node8(size + 1, elem, e0, e1, e2, e3, e4, e5, e6, next)
    else new Node8(1, elem, null, null, null, null, null, null, null, this)

  override def foreach(f: A => Unit): Unit =
    var current: List8[A] = this
    while current.isInstanceOf[Node8[?]] do
      val node = current.asInstanceOf[Node8[A]]
      val s = node.size
      f(node.e0)
      if s > 1 then f(node.e1.asInstanceOf[A])
      if s > 2 then f(node.e2.asInstanceOf[A])
      if s > 3 then f(node.e3.asInstanceOf[A])
      if s > 4 then f(node.e4.asInstanceOf[A])
      if s > 5 then f(node.e5.asInstanceOf[A])
      if s > 6 then f(node.e6.asInstanceOf[A])
      if s > 7 then f(node.e7.asInstanceOf[A])
      current = node.next

  override def ++[B >: A](that: List8[B]): List8[B] =
    if that.isEmpty then this
    else
      val first = new Node8[B](size, e0, e1, e2, e3, e4, e5, e6, e7, that)
      var last = first
      var current = next
      while current.isInstanceOf[Node8[?]] do
        val n = current.asInstanceOf[Node8[A]]
        val copied = new Node8[B](n.size, n.e0, n.e1, n.e2, n.e3, n.e4, n.e5, n.e6, n.e7, that)
        last.nextVar = copied
        last = copied
        current = n.next
      first

  // name-based extractor support: `case Node8(head, tail)` without Option/Tuple allocation
  def get: Node8[A] = this
  def _1: A = e0
  def _2: List8[A] = tail

object Node8:
  def unapply[A](node: Node8[A]): Node8[A] = node
