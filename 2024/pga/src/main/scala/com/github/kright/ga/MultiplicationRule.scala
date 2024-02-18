package com.github.kright.ga

class MultiplicationRule(using basis: Basis) extends HasBasis(basis):
  def dot(left: BasisBlade, right: BasisBlade): BasisBladeWithSign =
    val a = geometric(left, right)
    val b = geometric(right, left)
    if (a == b) a else BasisBladeWithSign(basis.scalarBlade, Sign.Zero)

  def wedge(left: BasisBlade, right: BasisBlade): BasisBladeWithSign =
    val a = geometric(left, right)
    val b = geometric(right, left)
    if (a == b) BasisBladeWithSign(basis.scalarBlade, Sign.Zero) else a

  def geometric(a: BasisBlade, b: BasisBlade): BasisBladeWithSign =
    checkBasis(a, b)

    val allBasisVectors = a.basisVectors ++ b.basisVectors
    val sorted = allBasisVectors.sorted

    val paritySign = if (Permutation.parity(allBasisVectors, sorted)) Sign.Positive else Sign.Negative
    val removedSigns = a.commonBasisVectors(b).map(_.getSquareSign)
    val sign = removedSigns.fold(paritySign)(_ * _)

    if (sign == Sign.Zero) return BasisBladeWithSign(basis.scalarBlade, sign)
    BasisBladeWithSign(BasisBlade(a.bits ^ b.bits), sign)

  private def checkBasis(left: HasBasis, right: HasBasis): Unit = {
    require(left.basis == basis)
    require(right.basis == basis)
  }
