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

  def geometric(a: BasisBladeWithSign, b: BasisBladeWithSign): BasisBladeWithSign =
    geometric(a.basisBlade, b.basisBlade) * (a.sign * b.sign)

  def rightComplement(a: BasisBlade): BasisBladeWithSign =
    val complement = a.anyComplement
    BasisBladeWithSign(complement, geometric(a, complement).sign)

  def rightComplement(a: BasisBladeWithSign): BasisBladeWithSign =
    rightComplement(a.basisBlade) * a.sign

  def leftComplement(a: BasisBlade): BasisBladeWithSign =
    val complement = a.anyComplement
    BasisBladeWithSign(complement, geometric(complement, a).sign)

  def leftComplement(a: BasisBladeWithSign): BasisBladeWithSign =
    leftComplement(a.basisBlade) * a.sign

  def geometricAntiproduct(a: BasisBlade, b: BasisBlade): BasisBladeWithSign =
    leftComplement(geometric(rightComplement(a), rightComplement(b)))

  private def checkBasis(left: HasBasis, right: HasBasis): Unit = {
    require(left.basis == basis)
    require(right.basis == basis)
  }
