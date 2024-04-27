package com.github.kright.ga

case class BasisVector(number: Int)(using basis: Basis) extends Ordered[BasisVector] with HasBasis(basis):
  require(number >= 0 && number < basis.signature.vectorsCount)

  def bitMask: Int = 1 << number

  override def compare(that: BasisVector): Int = {
    require(basis == that.basis)
    if (number > that.number) 1
    else if (number < that.number) -1
    else 0
  }

  def getSquareSign: Sign =
    if (number < basis.signature.pos) return Sign.Positive
    if (number < basis.signature.pos + basis.signature.neg) return Sign.Negative
    Sign.Zero

  override def toString: String = s"${basis.basisNames.names(number)}"

object BasisVector:
  def apply(name: Char)(using basis: Basis): BasisVector =
    BasisVector(basis.basisNames(name))
