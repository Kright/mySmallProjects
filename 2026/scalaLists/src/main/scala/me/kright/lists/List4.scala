package me.kright.lists

import scala.annotation.unchecked.uncheckedVariance

/**
 * Immutable single-linked list where each node stores up to 4 elements.
 * Elements are kept in fields (not in an array) to avoid an extra indirection.
 * Only e0 is guaranteed to be present, e1..e3 are typed as `A | Null`:
 * slots at indices >= size hold nulls.
 */
sealed trait List4[+A]:
  def isEmpty: Boolean

  def ::[B >: A](elem: B): List4[B]

  def foreach(f: A => Unit): Unit

  def ++[B >: A](that: List4[B]): List4[B]

  /** sums up the sizes of the nodes without touching the elements */
  def length: Int =
    var result = 0
    var current: List4[A] = this
    while current.isInstanceOf[Node4[?]] do
      val node = current.asInstanceOf[Node4[A]]
      result += node.size
      current = node.next
    result

object List4:
  def empty[A]: List4[A] = Nil4

  def apply[A](elems: A*): List4[A] =
    var result: List4[A] = Nil4
    var i = elems.length - 1
    while i >= 0 do
      result = elems(i) :: result
      i -= 1
    result

case object Nil4 extends List4[Nothing]:
  override def isEmpty: Boolean = true

  override def ::[B](elem: B): List4[B] =
    new Node4(1, elem, null, null, null, Nil4)

  override def foreach(f: Nothing => Unit): Unit = ()

  override def ++[B](that: List4[B]): List4[B] = that

// nextVar is mutable for the same reason as in scala.List:
// ++ builds copied nodes iteratively and links them in place, without recursion.
final class Node4[+A](val size: Int,
                      val e0: A,
                      val e1: A | Null, val e2: A | Null, val e3: A | Null,
                      private[lists] var nextVar: List4[A @uncheckedVariance]) extends List4[A]:
  def next: List4[A] = nextVar

  override def isEmpty: Boolean = false

  def head: A = e0

  /** the tail node is created dynamically: same elements shifted by one, without the head */
  def tail: List4[A] =
    if size == 1 then next
    else new Node4(size - 1, e1.asInstanceOf[A], e2, e3, null, next)

  override def ::[B >: A](elem: B): List4[B] =
    if size < 4 then new Node4(size + 1, elem, e0, e1, e2, next)
    else new Node4(1, elem, null, null, null, this)

  override def foreach(f: A => Unit): Unit =
    var current: List4[A] = this
    while current.isInstanceOf[Node4[?]] do
      val node = current.asInstanceOf[Node4[A]]
      val s = node.size
      f(node.e0)
      if s > 1 then f(node.e1.asInstanceOf[A])
      if s > 2 then f(node.e2.asInstanceOf[A])
      if s > 3 then f(node.e3.asInstanceOf[A])
      current = node.next

  override def ++[B >: A](that: List4[B]): List4[B] =
    if that.isEmpty then this
    else
      val first = new Node4[B](size, e0, e1, e2, e3, that)
      var last = first
      var current = next
      while current.isInstanceOf[Node4[?]] do
        val n = current.asInstanceOf[Node4[A]]
        val copied = new Node4[B](n.size, n.e0, n.e1, n.e2, n.e3, that)
        last.nextVar = copied
        last = copied
        current = n.next
      first

  // name-based extractor support: `case Node4(head, tail)` without Option/Tuple allocation
  def get: Node4[A] = this
  def _1: A = e0
  def _2: List4[A] = tail

object Node4:
  def unapply[A](node: Node4[A]): Node4[A] = node
