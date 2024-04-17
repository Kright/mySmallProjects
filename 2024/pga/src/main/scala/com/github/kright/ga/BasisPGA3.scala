package com.github.kright.ga

import scala.math.Numeric.Implicits.infixNumericOps

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

  def makeTranslate(nx: Double, ny: Double, nz: Double, shift: Double)(using basis: BasisPGA3): MultiVector[Double] = {
    val xyz = shift * 0.5
    val mult = 1.0 / Math.sqrt(nx * nx + ny * ny + nz * nz)
    val xyw = nz * mult
    val xzw = -ny * mult
    val yzw = nx * mult

    MultiVector(
      "xyz" -> shift * 0.5,
      "xyw" -> nz * mult,
      "xzw" -> -ny * mult,
      "yzw" -> nx * mult
    )
  }
  
  //todo method for making plane
