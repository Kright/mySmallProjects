package com.github.kright.ga

import com.github.kright.symbolic.SymbolicStr

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

  def plane[T](nx: T, ny: T, nz: T)(using num: Numeric[T], basis: BasisPGA3): MultiVector[T] =
    MultiVector(
      "xyw" -> nz,
      "xzw" -> -ny,
      "yzw" -> nx,
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


  def planeSymbolic(baseName: String)(using num: Numeric[SymbolicStr], basis: BasisPGA3): MultiVector[SymbolicStr] =
    MultiVector(
      "xyz" -> SymbolicStr(s"${baseName}.shift"),
      "xyw" -> SymbolicStr(s"${baseName}.nz"),
      "xzw" -> -SymbolicStr(s"${baseName}.ny"),
      "yzw" -> SymbolicStr(s"${baseName}.nx"),
    )

  def planeCenteredSymbolic(baseName: String)(using num: Numeric[SymbolicStr], basis: BasisPGA3): MultiVector[SymbolicStr] =
    MultiVector(
      "xyw" -> -SymbolicStr(s"${baseName}.nx"),
      "xzw" -> SymbolicStr(s"${baseName}.ny"),
      "yzw" -> SymbolicStr(s"${baseName}.nz"),
    )  

  def rotorSymbolic(baseName: String)(using num: Numeric[SymbolicStr], basis: BasisPGA3): MultiVector[SymbolicStr] =
    MultiVector(
        "xy" -> SymbolicStr(s"${baseName}.z"),
        "xz" -> SymbolicStr(s"${baseName}.y"),
        "yz" -> SymbolicStr(s"${baseName}.x"),
        "" -> SymbolicStr(s"${baseName}.s"),
    ).leftComplement // todo check signs
  
  def rotorSymbolic(xy: Double | String,
                    xz: Double | String,
                    yz: Double | String,
                    scalar: Double | String)(using num: Numeric[SymbolicStr], basis: BasisPGA3): MultiVector[SymbolicStr] =
    MultiVector(
      "xy" -> SymbolicStr(xy),
      "xz" -> SymbolicStr(xz),
      "yz" -> SymbolicStr(yz),
      "" -> SymbolicStr(scalar),
    ).leftComplement // todo check signs

  def translateSymbolic(baseName: String)(using num: Numeric[SymbolicStr], basis: BasisPGA3): MultiVector[SymbolicStr] =
    translateX2(
      SymbolicStr(0.5) * SymbolicStr(s"${baseName}.dx"),
      SymbolicStr(0.5) * SymbolicStr(s"${baseName}.dy"),
      SymbolicStr(0.5) * SymbolicStr(s"${baseName}.dz"),
    )

  def translateSymbolic(dx: Double | String, dy: Double | String, dz: Double | String)(using num: Numeric[SymbolicStr], basis: BasisPGA3): MultiVector[SymbolicStr] =
    translateX2(
      SymbolicStr(0.5) * SymbolicStr(dx),
      SymbolicStr(0.5) * SymbolicStr(dy),
      SymbolicStr(0.5) * SymbolicStr(dz),
    )  

  def pointSymbolic(baseName: String)(using num: Numeric[SymbolicStr], basis: BasisPGA3): MultiVector[SymbolicStr] =
    point[SymbolicStr](
      SymbolicStr(s"${baseName}.x"),
      SymbolicStr(s"${baseName}.y"),
      SymbolicStr(s"${baseName}.z"),
    )

  def vectorSymbolic(baseName: String)(using basis: BasisPGA3): MultiVector[SymbolicStr] =
    vector[SymbolicStr](
      SymbolicStr(s"${baseName}.x"),
      SymbolicStr(s"${baseName}.y"),
      SymbolicStr(s"${baseName}.z"),
    )
