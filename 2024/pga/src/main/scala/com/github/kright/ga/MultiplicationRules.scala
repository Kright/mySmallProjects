package com.github.kright.ga

class MultiplicationRules(using basis: Basis) extends HasBasis(basis):
  val geometric: Multiplication = (a: BasisBlade, b: BasisBlade) =>
    checkBasis(a, b)

    val allBasisVectors = a.basisVectors ++ b.basisVectors
    val sorted = allBasisVectors.sorted

    val paritySign = if (Permutation.parity(allBasisVectors, sorted)) Sign.Positive else Sign.Negative
    val removedSigns = a.commonBasisVectors(b).map(_.getSquareSign)
    val sign = removedSigns.fold(paritySign)(_ * _)

    if (sign == Sign.Zero) BasisBladeWithSign(basis.scalarBlade, sign)
    else BasisBladeWithSign(BasisBlade(a.bits ^ b.bits), sign)

  val dot: Multiplication = (left: BasisBlade, right: BasisBlade) =>
    if (left.hasCommonBasisVectors(right) || (left.bits == right.bits)) {
      geometric(left, right)
    } else {
      BasisBladeWithSign.zero
    }

  val wedge: Multiplication = (left: BasisBlade, right: BasisBlade) =>
    if (left.hasCommonBasisVectors(right) || (left.bits == right.bits)) {
      BasisBladeWithSign.zero
    } else {
      geometric(left, right)
    }

  val rightComplement: SingleOp = (a: BasisBlade) =>
    val complement = a.anyComplement
    BasisBladeWithSign(complement, geometric(a, complement).sign)

  val leftComplement: SingleOp = (a: BasisBlade) =>
    val complement = a.anyComplement
    BasisBladeWithSign(complement, geometric(complement, a).sign)

  val geometricAntiproduct: Multiplication = (a: BasisBlade, b: BasisBlade) =>
    leftComplement(geometric(rightComplement(a), rightComplement(b)))

  val wedgeAntiproduct: Multiplication = (a: BasisBlade, b: BasisBlade) =>
    leftComplement(wedge(rightComplement(a), rightComplement(b)))

  val dotAntiproduct: Multiplication = (a: BasisBlade, b: BasisBlade) =>
    leftComplement(dot(rightComplement(a), rightComplement(b)))

  private def isBulk(a: BasisBlade): Boolean =
    require(basis.neg == 0)
    geometric(a, a).sign != Sign.Zero

  val bulk: SingleOp = (a: BasisBlade) =>
    if (isBulk(a)) BasisBladeWithSign(a, Sign.Positive) else BasisBladeWithSign.zero

  val weight: SingleOp = (a: BasisBlade) =>
    if (isBulk(a)) BasisBladeWithSign.zero else BasisBladeWithSign(a, Sign.Positive)

  private def checkBasis(left: HasBasis, right: HasBasis): Unit = {
    require(left.basis == basis)
    require(right.basis == basis)
  }
