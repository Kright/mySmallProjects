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

  def plane(nx: Double, ny: Double, nz: Double, shift: Double)(using basis: BasisPGA3): MultiVector[Double] =
    val mult = 1.0 / Math.sqrt(nx * nx + ny * ny + nz * nz)
    val xyw = nz * mult
    val xzw = -ny * mult
    val yzw = nx * mult

    MultiVector(
      "xyz" -> shift,
      "xyw" -> nz * mult,
      "xzw" -> -ny * mult,
      "yzw" -> nx * mult
    ).withoutZeros

  def translateX2[T](halfDx: T, halfDy: T, halfDz: T)(using num: Numeric[T], basis: BasisPGA3): MultiVector[T] =
    MultiVector[T](
      "xy" -> halfDz,
      "xz" -> -halfDy,
      "yz" -> halfDx,
      "xyzw" -> num.one,
    ).withoutZeros

  def translate(dx: Double, dy: Double, dz: Double)(using basis: BasisPGA3): MultiVector[Double] =
    //      val r1 = makeReflection(nx, ny, nz, 0)
    //      val r2 = makeReflection(nx, ny, nz, len(vector(nx, ny, nz))
    //      r2.geometricAntiproduct(r1)
    translateX2(dx * 0.5, dy * 0.5, dz * 0.5)
  
  //todo method for making plane
