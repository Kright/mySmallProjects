package com.github.kright

trait FrequencyCounter[-T] extends (T => Int) {
  self =>

  def apply(t: T): Int
  def add(t: T): Unit
  def add(tt: IterableOnce[T]): Unit = tt.iterator.foreach(add)

  def map[U](f: U => T): FrequencyCounter[U] = new FrequencyCounter[U] {
    override def apply(t: U): Int = self(f(t))
    override def add(t: U): Unit = self.add(f(t))
    override def toString(): String = s"FrequencyCounter($self f)"
  }

  def mapOutput(f: Int => Int): FrequencyCounter[T] = new FrequencyCounter[T] {
    override def apply(t: T): Int = f(self(t))
    override def add(t: T): Unit = self.add(t)
    override def toString(): String = s"FrequencyCounter(f $self)"
  }
}


class SimpleFrequencyCounter[T] extends FrequencyCounter[T] {
  private val map = new mutable.HashMap[T, Int]()

  def add(t: T): Unit = {
    map(t) = 1 + map.getOrElse(t, 0)
  }

  def apply(t: T): Int =
    map.getOrElse(t, 0)

  override def toString(): String =
    s"SimpleFrequencyCounter(${map.size} unique, ${map.values.sum} total)"
}


object TotalFrequencyCounter {
  val words: FrequencyCounter[String] = new SimpleFrequencyCounter[String]()
}
