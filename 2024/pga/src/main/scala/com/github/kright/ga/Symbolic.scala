package com.github.kright.ga

sealed trait Symbolic:
  def prettyString: String

  override def toString: String = prettyString

  def isZero: Boolean =
    this match
      case Symbolic.Constant(0.0) => true
      case _ => false

  def isOne: Boolean =
    this match
      case Symbolic.Constant(1.0) => true
      case _ => false

  def isConstant: Boolean =
    this.isInstanceOf[Symbolic.Constant]

object Symbolic:
  case class Constant(value: Double) extends Symbolic:
    override def prettyString: String = value.toString

  object Constant:
    val zero = Constant(0.0)

  case class Symbol(name: String) extends Symbolic:
    override def prettyString: String = name

  case class Product(multiplier: Double, elems: Seq[Symbolic]) extends Symbolic:
    override def prettyString: String =
      if (multiplier == 1.0) return elems.map(_.prettyString).mkString(" * ")
      if (multiplier == -1.0) return "-" + elems.map(_.prettyString).mkString(" * ")
      (Seq(Constant(multiplier)) ++ elems).map(_.prettyString).mkString(" * ")



  object Product:
    def apply(elems: Seq[Symbolic]): Symbolic =
      val (multiplier, otherElems) = reduceElems(elems)

      if (multiplier == 0.0) return Constant.zero
      if (otherElems.isEmpty) return Constant(multiplier)
      Product(multiplier, otherElems)

    private def reduceElems(elems: Seq[Symbolic]): (Double, Seq[Symbolic]) =
      val (constants, otherElems) = elems.flatMap {
        case Symbolic.Product(mult, elems) => Seq(Constant(mult)) ++ elems
        case a: Symbolic => Seq(a)
      }.partition(_.isConstant)

      val product = constants.map(_.asInstanceOf[Symbolic.Constant].value).product
      (product, otherElems)

  case class Sum(elems: Seq[Symbolic]) extends Symbolic:
    override def prettyString: String = elems.map(_.prettyString).mkString(" + ")

  object Sum:
    def apply(elems: Seq[Symbolic]): Symbolic =
      val reducedElems = reduceElems(elems)

      reducedElems.size match
        case 0 => Constant(0.0)
        case 1 => reducedElems.head
        case _ => new Sum(reducedElems)

    private def reduceElems(elems: Seq[Symbolic]): Seq[Symbolic] =
      val (constants, otherElems) = elems.flatMap {
        case Symbolic.Sum(elems) => elems
        case a: Symbolic => Seq(a)
      }.partition(_.isConstant)

      val sum = constants.map(_.asInstanceOf[Symbolic.Constant].value).sum

      sum match
        case 0.0 => otherElems
        case p: Double => Seq(Symbolic.Constant(p)) ++ otherElems

  implicit val symbolicNumeric: Numeric[Symbolic] = new Numeric[Symbolic] {

    override def plus(x: Symbolic, y: Symbolic): Symbolic = Sum(Seq(x, y))

    override def minus(x: Symbolic, y: Symbolic): Symbolic = Sum(Seq(x, Product(Seq[Symbolic](Constant(-1.0), y))))

    override def times(x: Symbolic, y: Symbolic): Symbolic = Product(Seq(x, y))

    override def negate(x: Symbolic): Symbolic = Product(Seq(Constant(-1.0), x))

    override def fromInt(x: Int): Symbolic = Constant(x)

    override def parseString(str: String): Option[Symbolic] = ???

    override def toInt(x: Symbolic): Int = ???

    override def toLong(x: Symbolic): Long = ???

    override def toFloat(x: Symbolic): Float = ???

    override def toDouble(x: Symbolic): Double = ???

    override def compare(x: Symbolic, y: Symbolic): Int = ???
  }