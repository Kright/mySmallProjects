package com.github.kright.symbolic


sealed trait Symbolic:
  def isZero: Boolean =
    this match
      case Constant(0.0) => true
      case _ => false

  def isOne: Boolean =
    this match
      case Constant(1.0) => true
      case _ => false

  def isFunc: Boolean =
    this match
      case f: Func => true
      case _ => false

  def isFunc(name: String): Boolean =
    this match
      case f: Func => f.name == name
      case _ => false


case class Constant(value: Double) extends Symbolic:
  override def toString: String = value.toString

object Constant:
  val zero = Constant(0.0)
  val one = Constant(1.0)

case class Symbol(name: String) extends Symbolic:
  override def toString: String = name

case class Func(name: String, elems: Seq[Symbolic]) extends Symbolic:
  override def toString: String = s"${name}(${elems.mkString(", ")})"


object Symbolic:
  implicit val symbolicNumeric: Numeric[Symbolic] = new Numeric[Symbolic] {

    override def plus(x: Symbolic, y: Symbolic): Symbolic = Func("+", Seq(x, y))

    override def minus(x: Symbolic, y: Symbolic): Symbolic = Func("-", Seq(x, y))

    override def times(x: Symbolic, y: Symbolic): Symbolic = Func("*", Seq(x, y))

    override def negate(x: Symbolic): Symbolic = Func("-", Seq(x))

    override def fromInt(x: Int): Symbolic = Constant(x)

    override def parseString(str: String): Option[Symbolic] = ???

    override def toInt(x: Symbolic): Int = ???

    override def toLong(x: Symbolic): Long = ???

    override def toFloat(x: Symbolic): Float = ???

    override def toDouble(x: Symbolic): Double = ???

    override def compare(x: Symbolic, y: Symbolic): Int = ???
  }

  extension (s: Symbolic)
    def map(f: Symbolic => Symbolic): Symbolic =
      s match
        case c: Constant => f(c)
        case s: Symbol => f(s)
        case Func(name, elems) => f(Func(name, elems.map(f)))