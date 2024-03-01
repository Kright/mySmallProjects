package com.github.kright.ga

sealed trait Symbolic:
  def prettyString: String

  override def toString: String = prettyString

  def simplified: Symbolic =
    this match
      case Symbolic.Constant(value) => this
      case Symbolic.Symbol(s) => this
      case Symbolic.Product(elems) => {
        val (constants, otherElems) = elems.map(_.simplified).partition(_.isConstant)
        val multiplier = constants.map(_.asInstanceOf[Symbolic.Constant].value).product
        multiplier match
          case 0.0 => Symbolic.Constant(0.0)
          case 1.0 => Symbolic.Product.optimized(otherElems)
          case v: Double => Symbolic.Product.optimized(Seq(Symbolic.Constant(v)) ++ otherElems)
      }
      case Symbolic.Sum(elems) => {
        val (constants, otherElems) = elems.map(_.simplified).partition(_.isConstant)
        val sum = constants.map(_.asInstanceOf[Symbolic.Constant].value).sum
        sum match
          case 0.0 => Symbolic.Sum.optimized(otherElems)
          case v => Symbolic.Sum.optimized(Seq(Symbolic.Constant(v)) ++ otherElems)
      }

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

  case class Symbol(name: String) extends Symbolic:
    override def prettyString: String = name

  case class Product(elems: Seq[Symbolic]) extends Symbolic:
    override def prettyString: String = elems.map(_.prettyString).mkString(" * ")

  object Product:
    def optimized(v: Seq[Symbolic]): Symbolic =
      v.size match
        case 0 => Constant(1.0)
        case 1 => v.head
        case _ => Product(v)

  case class Sum(elems: Seq[Symbolic]) extends Symbolic:
    override def prettyString: String = elems.map(_.prettyString).mkString(" + ")

  case object Sum:
    def optimized(v: Seq[Symbolic]): Symbolic =
      v.size match
        case 0 => Constant(0.0)
        case 1 => v.head
        case _ => Sum(v)


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