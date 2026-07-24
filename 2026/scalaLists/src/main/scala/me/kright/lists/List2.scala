package me.kright.lists

import scala.annotation.unchecked.uncheckedVariance

/**
 * Immutable single-linked list where each node stores up to 2 elements.
 * Elements are kept in fields (not in an array) to avoid an extra indirection.
 * Only e0 is guaranteed to be present, e1 is typed as `A | Null`:
 * slots at indices >= size hold nulls.
 */
sealed trait List2[+A]:
  def isEmpty: Boolean

  def ::[B >: A](elem: B): List2[B]

  def foreach(f: A => Unit): Unit

  def ++[B >: A](that: List2[B]): List2[B]

  /** sums up the sizes of the nodes without touching the elements */
  def length: Int =
    var result = 0
    var current: List2[A] = this
    while current.isInstanceOf[Node2[?]] do
      val node = current.asInstanceOf[Node2[A]]
      result += node.size
      current = node.next
    result

object List2:
  def empty[A]: List2[A] = Nil2

  def apply[A](elems: A*): List2[A] =
    var result: List2[A] = Nil2
    var i = elems.length - 1
    while i >= 0 do
      result = elems(i) :: result
      i -= 1
    result

case object Nil2 extends List2[Nothing]:
  override def isEmpty: Boolean = true

  override def ::[B](elem: B): List2[B] =
    new Node2(1, elem, null, Nil2)

  override def foreach(f: Nothing => Unit): Unit = ()

  override def ++[B](that: List2[B]): List2[B] = that

// nextVar is mutable for the same reason as in scala.List:
// ++ builds copied nodes iteratively and links them in place, without recursion.
final class Node2[+A](val size: Int,
                      val e0: A, val e1: A | Null,
                      private[lists] var nextVar: List2[A @uncheckedVariance]) extends List2[A]:
  def next: List2[A] = nextVar

  override def isEmpty: Boolean = false

  def head: A = e0

  /** the tail node is created dynamically: the same elements without the head */
  def tail: List2[A] =
    if size == 1 then next
    else new Node2(size - 1, e1.asInstanceOf[A], null, next)

  override def ::[B >: A](elem: B): List2[B] =
    if size < 2 then new Node2(size + 1, elem, e0, next)
    else new Node2(1, elem, null, this)

  override def foreach(f: A => Unit): Unit =
    var current: List2[A] = this
    while current.isInstanceOf[Node2[?]] do
      val node = current.asInstanceOf[Node2[A]]
      f(node.e0)
      if node.size > 1 then f(node.e1.asInstanceOf[A])
      current = node.next

  override def ++[B >: A](that: List2[B]): List2[B] =
    if that.isEmpty then this
    else
      val first = new Node2[B](size, e0, e1, that)
      var last = first
      var current = next
      while current.isInstanceOf[Node2[?]] do
        val n = current.asInstanceOf[Node2[A]]
        val copied = new Node2[B](n.size, n.e0, n.e1, that)
        last.nextVar = copied
        last = copied
        current = n.next
      first

  // name-based extractor support: `case Node2(head, tail)` without Option/Tuple allocation
  def get: Node2[A] = this
  def _1: A = e0
  def _2: List2[A] = tail

object Node2:
  def unapply[A](node: Node2[A]): Node2[A] = node
