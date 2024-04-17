package com.github.kright.ga

import scala.math.Numeric.Implicits.infixNumericOps


class BasisPGA2(basisNames: BasisNames) extends Basis(2, 0, 1, basisNames)

object BasisPGA2:
  def point[T](x: T, y: T, w: T)(using basis: BasisPGA2): MultiVector[T] =
    MultiVector[T](
      "x" -> x,
      "y" -> y,
      "w" -> w,
    )

  def point[T](x: T, y: T)(using num: Numeric[T], basis: BasisPGA2): MultiVector[T] =
    point(x, y, num.one)

  def vector[T](x: T, y: T)(using basis: BasisPGA2): MultiVector[T] =
    MultiVector[T](
      "x" -> x,
      "y" -> y,
    )

  def translate[T](halfDx: T, halfDy: T)(using num: Numeric[T], basis: BasisPGA2): MultiVector[T] =
//    val perpDx = -halfDy
//    val perpDy = halfDx
//    val centerLine = point(num.zero, num.zero).wedge(point(perpDx, perpDy))
//    val shiftedLine = point(halfDx, halfDy).wedge(point(perpDx + halfDx, perpDy + halfDy))
//    val result = shiftedLine ⟇ centerLine
//    result.withoutZeros
    MultiVector(
      "x" -> halfDx,
      "y" -> -halfDy,
      "xyw" -> num.one,
    ).withoutZeros

  def rotate[T](cos2a: T, sin2a: T)(using num: Numeric[T], basis: BasisPGA2): MultiVector[T] =
    val centerLine = point(num.zero, num.zero).wedge(point(num.one, num.zero))
    val rotatedLine = point(num.zero, num.zero).wedge(point(cos2a, sin2a))
    val result = rotatedLine ⟇ centerLine
    result.withoutZeros

//  def motor()  