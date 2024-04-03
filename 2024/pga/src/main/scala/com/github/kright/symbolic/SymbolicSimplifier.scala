package com.github.kright.symbolic


object SymbolicSimplifier:
  def apply(maxRepeatCount: Int = 64): SymbolicTransformRepeater =
    SymbolicTransformAny(allSimplifiers()).repeat(maxRepeatCount)

  def allSimplifiers(): Seq[SymbolicPartialTransform] = Seq(
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

  class SumZeroRemover extends SymbolicRecursiveTransformHelper:
    override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
      symbolic match
        case Func("+", elems) if elems.exists(_.isZero) => Option(sumOrElement(elems.filterNot(_.isZero)))
        case _ => None

    private def sumOrElement(elements: Seq[Symbolic]): Symbolic =
      elements.size match
        case 0 => Constant.zero
        case 1 => elements.head
        case 2 => Func("+", elements)

  class BinarySubToUnary extends SymbolicRecursiveTransformHelper:
    override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
      symbolic match
        case Func("-", elems) if elems.size == 2 => Option {
          if (elems(1).isZero) elems.head
          else if (elems(0).isZero) Func("-", Seq(elems(1)))
          else Func("+", Seq(elems(0), Func("-", Seq(elems(1)))))
        }
        case _ => None

  class ProductOneRemover extends SymbolicRecursiveTransformHelper:
    override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
      symbolic match
        case Func("*", elems) if elems.exists(_.isOne) => Option(sumOrElement(elems.filterNot(_.isOne)))
        case _ => None

    private def sumOrElement(elements: Seq[Symbolic]): Symbolic =
      elements.size match
        case 0 => Constant.one
        case 1 => elements.head
        case 2 => Func("*", elements)

  class ProductZeroRemover extends SymbolicRecursiveTransformHelper:
    override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
      symbolic match
        case Func("*", elems) if elems.exists(_.isZero) => Option(Constant.zero)
        case _ => None

  class SumFlattener extends SymbolicRecursiveTransformHelper:
    override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
      symbolic match
        case Func("+", elems) if elems.exists {
          case Func("+", elems) => true
          case _ => false
        } => Option {
          Func("+", elems.flatMap {
            case Func("+", elems) => elems
            case other => Seq(other)
          })
        }
        case _ => None

  class SumWithMinusSumFlattener extends SymbolicRecursiveTransformHelper:
    override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
      symbolic match
        case Func("+", elems) if elems.exists {
          case Func("-", Seq(Func("+", elems))) => true
          case _ => false
        } => Option {
          Func("+", elems.flatMap {
            case Func("-", Seq(Func("+", innerElems))) => innerElems.map(e => Func("-", Seq(e)))
            case other => Seq(other)
          })
        }
        case _ => None

  class ProductFlattener extends SymbolicRecursiveTransformHelper:
    override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
      symbolic match
        case Func("*", elems) if elems.exists {
          case Func("*", elems) => true
          case _ => false
        } => Option {
          Func("*", elems.flatMap {
            case Func("*", elems) => elems
            case other => Seq(other)
          })
        }
        case _ => None

  class ProductOfSumToSumOfProducts extends SymbolicRecursiveTransformHelper:
    override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
      symbolic match
        case Func("*", elems) if elems.exists(_.isFunc("+")) => {
          val (sums, others) = elems.partition(_.isFunc("+"))
          val Func("+", topSumElems) = sums.head
          val tail = sums.tail ++ others
          Option(Func("+", topSumElems.map(el => Func("*", Seq(el) ++ tail))))
        }
        case _ => None

  class DoubleMinusRemover extends SymbolicRecursiveTransformHelper:
    override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
      symbolic match
        case Func("-", Seq(Func("-", Seq(elem)))) => Option(elem)
        case _ => None

  class MinusToTheUpOfProduct extends SymbolicRecursiveTransformHelper:
    override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
      symbolic match
        case Func("*", elems) if elems.exists(isUnaryMinus) => Option {
          var minusedCount = 0
          val resultWithoutSign = Func("*", elems.map {
            case Func("-", elems) if elems.size == 1 =>
              minusedCount += 1
              elems.head
            case other => other
          })
          if (minusedCount % 2 == 0) resultWithoutSign
          else Func("-", Seq(resultWithoutSign))
        }
        case _ => None

    private def isUnaryMinus(s: Symbolic): Boolean =
      s match
        case Func("-", elems) if elems.size == 1 => true
        case _ => false
