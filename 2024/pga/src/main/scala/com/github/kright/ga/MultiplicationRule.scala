package com.github.kright.ga

class MultiplicationRule(using basis: Basis) extends HasBasis(basis):
  def dot(left: BasisBlade, right: BasisBlade): BasisBladeWithSign =
    if (left.hasCommonBasisVectors(right) || (left.bits == right.bits)) {
      geometric(left, right)
    } else {
      BasisBladeWithSign(basis.scalarBlade, Sign.Zero)
    }

  def wedge(left: BasisBlade, right: BasisBlade): BasisBladeWithSign =
    if (left.hasCommonBasisVectors(right) || (left.bits == right.bits)) {
      BasisBladeWithSign(basis.scalarBlade, Sign.Zero)
    } else {
      geometric(left, right)
    }

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
