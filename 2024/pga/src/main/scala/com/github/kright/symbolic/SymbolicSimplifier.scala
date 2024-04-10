package com.github.kright.symbolic


object SymbolicSimplifier:
  def apply(maxRepeatCount: Int = 64): SymbolicTransformRepeater[SimpleSymbolic.Func, SimpleSymbolic.Symbol] =
    SymbolicTransformAny(allSimplifiers()).repeat(maxRepeatCount)

  def allSimplifiers(): Seq[SymbolicPartialTransform[SimpleSymbolic.Func, SimpleSymbolic.Symbol]] = Seq(
    BinarySubToUnary(),
    SumZeroRemover(),
    ProductOneRemover(),
    ProductZeroRemover(),
    SumFlattener(),
    ProductFlattener(),
    ProductOfSumToSumOfProducts(),
    MinusToTheUpOfProduct(),
    SumWithMinusSumFlattener(),
    DoubleMinusRemover(),
  )
  
  type TransformHelper = SymbolicRecursiveTransformHelper[SimpleSymbolic.Func, SimpleSymbolic.Symbol]

  class SumZeroRemover extends TransformHelper:
    override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
      symbolic match
        case Symbolic.Func("+", elems) if elems.exists(_.isZero) => Option(sumOrElement(elems.filterNot(_.isZero)))
        case _ => None

    private def sumOrElement(elements: Seq[SimpleSymbolic]): SimpleSymbolic =
      elements.size match
        case 0 => SimpleSymbolic.zero
        case 1 => elements.head
        case 2 => SimpleSymbolic("+", elements)

  class BinarySubToUnary extends TransformHelper:
    override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
      symbolic match
        case Symbolic.Func("-", elems) if elems.size == 2 => Option {
          if (elems(1).isZero) elems.head
          else if (elems(0).isZero) Symbolic.Func("-", Seq(elems(1)))
          else Symbolic.Func("+", Seq(elems(0), Symbolic.Func("-", Seq(elems(1)))))
        }
        case _ => None

  class ProductOneRemover extends TransformHelper:
    override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
      symbolic match
        case Symbolic.Func("*", elems) if elems.exists(_.isOne) => Option(sumOrElement(elems.filterNot(_.isOne)))
        case _ => None

    private def sumOrElement(elements: Seq[SimpleSymbolic]): SimpleSymbolic =
      elements.size match
        case 0 => SimpleSymbolic.one
        case 1 => elements.head
        case 2 => Symbolic.Func("*", elements)

  class ProductZeroRemover extends TransformHelper:
    override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
      symbolic match
        case Symbolic.Func("*", elems) if elems.exists(_.isZero) => Option(SimpleSymbolic.zero)
        case _ => None

  class SumFlattener extends TransformHelper:
    override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
      symbolic match
        case Symbolic.Func("+", elems) if elems.exists {
          case Symbolic.Func("+", elems) => true
          case _ => false
        } => Option {
          Symbolic.Func("+", elems.flatMap {
            case Symbolic.Func("+", elems) => elems
            case other => Seq(other)
          })
        }
        case _ => None

  class SumWithMinusSumFlattener extends TransformHelper:
    override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
      symbolic match
        case Symbolic.Func("+", elems) if elems.exists {
          case Symbolic.Func("-", Seq(Symbolic.Func("+", elems))) => true
          case _ => false
        } => Option {
          Symbolic.Func("+", elems.flatMap {
            case Symbolic.Func("-", Seq(Symbolic.Func("+", innerElems))) => innerElems.map(e => Symbolic.Func("-", Seq(e)))
            case other => Seq(other)
          })
        }
        case _ => None

  class ProductFlattener extends TransformHelper:
    override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
      symbolic match
        case Symbolic.Func("*", elems) if elems.exists {
          case Symbolic.Func("*", elems) => true
          case _ => false
        } => Option {
          Symbolic.Func("*", elems.flatMap {
            case Symbolic.Func("*", elems) => elems
            case other => Seq(other)
          })
        }
        case _ => None

  class ProductOfSumToSumOfProducts extends TransformHelper:
    override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
      symbolic match
        case Symbolic.Func("*", elems) if elems.exists(_.isFunc("+")) => {
          val (sums, others) = elems.partition(_.isFunc("+"))
          val Symbolic.Func("+", topSumElems) = sums.head
          val tail = sums.tail ++ others
          Option(Symbolic.Func("+", topSumElems.map(el => Symbolic.Func("*", Seq(el) ++ tail))))
        }
        case _ => None

  class DoubleMinusRemover extends TransformHelper:
    override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
      symbolic match
        case Symbolic.Func("-", Seq(Symbolic.Func("-", Seq(elem)))) => Option(elem)
        case _ => None

  class MinusToTheUpOfProduct extends TransformHelper:
    override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
      symbolic match
        case Symbolic.Func("*", elems) if elems.exists(isUnaryMinus) => Option {
          var minusedCount = 0
          val resultWithoutSign = Symbolic.Func("*", elems.map {
            case Symbolic.Func("-", elems) if elems.size == 1 =>
              minusedCount += 1
              elems.head
            case other => other
          })
          if (minusedCount % 2 == 0) resultWithoutSign
          else Symbolic.Func("-", Seq(resultWithoutSign))
        }
        case _ => None

    private def isUnaryMinus(s: SimpleSymbolic): Boolean =
      s match
        case Symbolic.Func("-", elems) if elems.size == 1 => true
        case _ => false
