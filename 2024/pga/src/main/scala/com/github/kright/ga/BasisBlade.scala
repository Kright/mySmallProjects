package com.github.kright.ga

case class BasisBlade(bits: Int)(using basis: Basis) extends HasBasis(basis):
  require(bits >= 0 && bits < basis.bladesCount)

  def contains(v: BasisVector): Boolean =
    require(v.basis == this.basis)
    (bits & v.bitMask) != 0

  def basisVectors: Seq[BasisVector] =
    basis.vectors.filter(contains)

  def order: Int =
    Integer.bitCount(bits)

  def commonBasisVectors(other: BasisBlade): Seq[BasisVector] =
    require(basis == other.basis)
    BasisBlade(bits & other.bits).basisVectors

  def hasCommonBasisVectors(other: BasisBlade): Boolean =
    (bits & other.bits) != 0

  /* doesnt account sign */
  def anyComplement: BasisBlade =
    BasisBlade(bits ^ basis.bitsMap)

  override def toString: String =
    if (bits == 0) return "1"
    if (bits == (1 << basis.vectorsCount) - 1) return "I"
    s"${basisVectors.map(v => basis.basisNames.names(v.number)).mkString("")}"


object BasisBlade:
  /**
   * Unaware of order, so "xy" and "yx" will produce BasisBlade("xy") without sign checking
   * Repeating same symbol is not allowed
   */
  def apply(names: String)(using basis: Basis): BasisBlade =
    require(names.distinct.length == names.length)
    BasisBlade(names.map(1 << basis.basisNames(_)).fold(0)(_ | _))

  def apply(basisVector: BasisVector)(using basis: Basis): BasisBlade =
    BasisBlade(basisVector.bitMask)