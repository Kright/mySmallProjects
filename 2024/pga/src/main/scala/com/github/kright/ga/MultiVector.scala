package com.github.kright.ga

import com.github.kright.symbolic.SymbolicStr.given
import com.github.kright.symbolic.transform.simplifiers.SymbolicStrSimplifier
import com.github.kright.symbolic.{SymbolicStr, SymbolicToPrettyString}

import scala.collection.mutable
import scala.math.Numeric.Implicits.infixNumericOps

case class MultiVector[Value](values: Map[BasisBlade, Value])(using basis: Basis) extends HasBasis(basis):

  def apply(b: BasisBlade)(using num: Numeric[Value]): Value =
    values.getOrElse(b, num.zero)

  def map[V2](f: (BasisBlade, Value) => V2): MultiVector[V2] =
    MultiVector[V2](values.map((b, v) => b -> f(b, v)))

  def mapValues[V2](f: Value => V2): MultiVector[V2] =
    MultiVector[V2](values.map((b, v) => b -> f(v)))

  def filter(f: (BasisBlade, Value) => Boolean): MultiVector[Value] =
    MultiVector[Value](values.filter(f(_, _)))

  def updated(setValues: IterableOnce[(BasisBlade, Value)]): MultiVector[Value] =
    MultiVector[Value](values ++ setValues)

  override def toString: String =
    toString("(", ", ", ")")

  def toString(start: String, separator: String, end: String): String =
    if (values.isEmpty)
      "MultiVector()"
    else
      s"MultiVector${values.toSeq.sortWith((p1, p2) => (p1._1.grade < p2._1.grade) || (p1._1.bits < p2._1.bits)).map { (b, v) => s"$b -> ${v}" }.mkString(start, separator, end)}"

  def toMultilineString: String =
    toString("(\n", "\n", "\n)")

object MultiVector:
  def apply(b: BasisBlade)(using basis: Basis): MultiVector[Double] =
    new MultiVector[Double](Map(b -> 1.0))

  def apply[T](values: IterableOnce[(BasisBlade, T)])(using basis: Basis): MultiVector[T] =
    new MultiVector[T](values.iterator.toMap)

  def apply[T](values: (String, T)*)(using basis: Basis): MultiVector[T] =
    apply[T](values.map((s, v) => BasisBlade(s) -> v).toMap)

  def scalar[T](value: T)(using basis: Basis): MultiVector[Double] =
    new MultiVector[Double](Map(basis.scalarBlade -> 1.0))

  def zero[T](using basis: Basis): MultiVector[T] =
    new MultiVector[T](Map.empty[BasisBlade, T])

  def randomDoubles(using basis: Basis): MultiVector[Double] =
    apply(basis.blades.map(b => b -> math.random()))

  extension [T](left: MultiVector[T])(using num: Numeric[T])
    def multiply(right: MultiVector[T], mult: MultiplicationTable): MultiVector[T] =
      require(left.basis == right.basis)
      val result = new mutable.HashMap[BasisBlade, T]()
      for ((l, lv) <- left.values) {
        for ((r, rv) <- right.values) {
          val rr = mult(l, r)
          rr.sign match
            case Sign.Positive => result(rr.basisBlade) = result.getOrElse(rr.basisBlade, num.zero) + lv * rv
            case Sign.Negative => result(rr.basisBlade) = result.getOrElse(rr.basisBlade, num.zero) - lv * rv
            case Sign.Zero =>
        }
      }
      new MultiVector[T](result.toMap)(using left.basis)

    def applySingleOp(singleOp: SingleOpTable): MultiVector[T] = {
      val result = new mutable.HashMap[BasisBlade, T]()
      for ((l, lv) <- left.values) {
        val rr = singleOp(l)
        rr.sign match
          case Sign.Positive => result(rr.basisBlade) = result.getOrElse(rr.basisBlade, num.zero) + lv
          case Sign.Negative => result(rr.basisBlade) = result.getOrElse(rr.basisBlade, num.zero) - lv
          case Sign.Zero =>
      }
      new MultiVector[T](result.toMap)(using left.basis)
    }

    infix def geometric(right: MultiVector[T]): MultiVector[T] = multiply(right, left.basis.geometric)
    infix def wedge(right: MultiVector[T]): MultiVector[T] = multiply(right, left.basis.wedge)
    infix def dot(right: MultiVector[T]): MultiVector[T] = multiply(right, left.basis.dot)

    infix def antiGeometric(right: MultiVector[T]): MultiVector[T] = multiply(right, left.basis.antiGeometric)
    infix def antiWedge(right: MultiVector[T]): MultiVector[T] = multiply(right, left.basis.antiWedge)
    infix def antiDot(right: MultiVector[T]): MultiVector[T] = multiply(right, left.basis.antiDot)

    infix def sandwich(middle: MultiVector[T]): MultiVector[T] = left.geometric(middle).geometric(left.reverse)
    infix def antiSandwich(middle: MultiVector[T]): MultiVector[T] = left.antiGeometric(middle).antiGeometric(left.antiReverse)

    def rightComplement: MultiVector[T] = applySingleOp(left.basis.rightComplement)
    def leftComplement: MultiVector[T] = applySingleOp(left.basis.leftComplement)
    def bulk: MultiVector[T] = applySingleOp(left.basis.bulk)
    def weight: MultiVector[T] = applySingleOp(left.basis.weight)
    def reverse: MultiVector[T] = applySingleOp(left.basis.reverse)
    def antiReverse: MultiVector[T] = applySingleOp(left.basis.antiReverse)

    // unicode symbols: https://projectivegeometricalgebra.org/
    infix def ⟑(right: MultiVector[T]): MultiVector[T] = geometric(right)
    infix def ∧(right: MultiVector[T]): MultiVector[T] = wedge(right)
    infix def ⋅(right: MultiVector[T]): MultiVector[T] = dot(right)

    infix def ⟇(right: MultiVector[T]): MultiVector[T] = antiGeometric(right)
    infix def ∨(right: MultiVector[T]): MultiVector[T] = antiWedge(right)
    infix def ◦(right: MultiVector[T]): MultiVector[T] = antiDot(right)

    infix def +(right: MultiVector[T]): MultiVector[T] =
      MultiVector[T]((left.values.keySet ++ right.values.keySet).toSeq.map { b =>
        b -> (left(b) + right(b))
      })(using left.basis)

    infix def -(right: MultiVector[T]): MultiVector[T] =
      MultiVector[T]((left.values.keySet ++ right.values.keySet).toSeq.map { b =>
        b -> (left(b) - right(b))
      })(using left.basis)

    def unary_- : MultiVector[T] =
      left.map((b, v) => -v)

    infix def *(scalarMultiplier: T): MultiVector[T] =
      left.map((b, t) => t * scalarMultiplier)

    def apply(right: MultiVector[T]): MultiVector[T] = geometric(right)

    def getSqrDist(right: MultiVector[T]): T =
      (left.values.keySet ++ right.values.keySet).view.map { b =>
        val d = left(b) - right(b)
        d * d
      }.sum

    def squareMagnitude: T =
      left.values.values.map(v => v * v).sum

    def withoutZeros: MultiVector[T] =
      left.filter((_, v) => v != num.zero)

  extension (left: MultiVector[Double])
    def magnitude: Double =
      math.sqrt(left.squareMagnitude)

    def /(scalarDivider: Double): MultiVector[Double] =
      left * (1.0 / scalarDivider)

    def norm: Double =
      math.sqrt(left.values.values.map(v => v * v).sum)

    def normalizedByWeight: MultiVector[Double] =
      left / left.weight.norm

    def normalizedByNorm: MultiVector[Double] =
      left / left.norm

  private val symbolicSimplify = SymbolicStrSimplifier.simplify(maxRepeatCount = 64)

  extension (v: MultiVector[SymbolicStr])
    def toPrettyMultilineString =
      v.mapValues(symbolicSimplify.transform)
        .withoutZeros
        .mapValues(SymbolicToPrettyString(_))
        .toMultilineString
