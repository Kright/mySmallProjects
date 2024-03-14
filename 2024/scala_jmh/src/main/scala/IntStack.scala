package com.github.kright

final class IntStack(size: Int) {
  val array = new Array[Int](size)
  var sPointer = -1

  def push(v: Int): Unit =
    sPointer += 1
    array(sPointer) = v
    
  def add(): Unit =
    array(sPointer - 1) += array(sPointer)
    sPointer -= 1

  def pop(): Int =
    val value = array(sPointer)
    sPointer -= 1
    value

  def head: Int = array(sPointer)

  def head_=(v: Int): Unit =
    array(sPointer) = v

  def get(shift: Int): Int =
    array(sPointer - shift)

  def dup(shift: Int): Unit =
    push(get(shift))
    
  def copyHead(shift: Int): Unit =
    array(sPointer - shift) = array(sPointer)  
}
