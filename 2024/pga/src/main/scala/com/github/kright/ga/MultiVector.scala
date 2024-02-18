package com.github.kright.ga

import scala.collection.mutable
import scala.math.Numeric.Implicits.infixNumericOps

case class MultiVector[Value](values: Map[BasisBlade, Value])(using basis: Basis) extends HasBasis(basis):

  def apply(b: BasisBlade)(using num: Numeric[Value]): Value =
    values.getOrElse(b, num.zero)

  def map[V2](f: (BasisBlade, Value) => V2): MultiVector[V2] =
    MultiVector[V2](values.map((b, v) => b -> f(b, v)))

  def filter(f: (BasisBlade, Value) => Boolean): MultiVector[Value] =
    MultiVector[Value](values.filter(f(_, _)))

  def updated(setValues: IterableOnce[(BasisBlade, Value)]): MultiVector[Value] =
    MultiVector[Value](values ++ setValues)

  override def toString: String =
    s"MultiVector(${values.toSeq.sortWith((p1, p2) => (p1._1.order < p2._1.order) || (p1._1.bits < p2._1.bits)).map { (b, v) => s"$b -> ${v}" }.mkString(", ")})"


object MultiVector:
  def apply(b: BasisBlade)(using basis: Basis): MultiVector[Double] =
    new MultiVector[Double](Map(b -> 1.0))

  def apply[T](values: IterableOnce[(BasisBlade, T)])(using basis: Basis): MultiVector[T] =
    new MultiVector[T](values.iterator.toMap)

  def scalar[T](value: T)(using basis: Basis): MultiVector[Double] =
    new MultiVector[Double](Map(basis.scalarBlade -> 1.0))

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

    def geometric(right: MultiVector[T]): MultiVector[T] = multiply(right, left.basis.geometric)
    def wedge(right: MultiVector[T]): MultiVector[T] = multiply(right, left.basis.wedge)

    def dot(right: MultiVector[T]): MultiVector[T] = multiply(right, left.basis.dot)

    // unicode symbols: https://projectivegeometricalgebra.org/
    infix def ⟑(right: MultiVector[T]): MultiVector[T] = geometric(right)
    def ∧(right: MultiVector[T]): MultiVector[T] = wedge(right)
    def ⋅(right: MultiVector[T]): MultiVector[T] = dot(right)

    def +(right: MultiVector[T]): MultiVector[T] =
      MultiVector[T]((left.values.keySet ++ right.values.keySet).toSeq.map { b =>
        b -> (left(b) + right(b))
      })(using left.basis)

    def -(right: MultiVector[T]): MultiVector[T] =
      MultiVector[T]((left.values.keySet ++ right.values.keySet).toSeq.map { b =>
        b -> (left(b) - right(b))
      })(using left.basis)

    def unary_- : MultiVector[T] =
      left.map((b, v) => -v)

    def apply(right: MultiVector[T]): MultiVector[T] = geometric(right)

    def getSqrDist(right: MultiVector[T]): T =
      (left.values.keySet ++ right.values.keySet).view.map { b =>
        val d = left(b) - right(b)
        d * d
      }.sum

    def squareMagnitude: T =
      left.values.values.map(v => v * v).sum

  extension (left: MultiVector[Double])
    def magnitude: Double =
      math.sqrt(left.squareMagnitude)