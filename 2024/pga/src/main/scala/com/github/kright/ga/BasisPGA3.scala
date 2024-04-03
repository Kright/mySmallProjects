package com.github.kright.ga

class BasisPGA3(basisNames: BasisNames) extends Basis(3, 0, 1, basisNames)

object BasisPGA3:
  def point[T](x: T, y: T, z: T, w: T)(using basis: BasisPGA3): MultiVector[T] =
    MultiVector[T](
      "x" -> x,
      "y" -> y,
      "z" -> z,
      "w" -> w,
    )

  def point[T](x: T, y: T, z: T)(using num: Numeric[T], basis: BasisPGA3): MultiVector[T] =
    point(x, y, z, num.one)

  def vector[T](x: T, y: T, z: T)(using basis: BasisPGA3): MultiVector[T] =
    MultiVector[T](
      "x" -> x,
      "y" -> y,
      "z" -> z,
    )
