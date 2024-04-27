package com.github.kright.ga


class CachedMultiplication(private val rule: Multiplication, 
                           private val basis: Basis) extends Multiplication:
  private val data = new Array[BasisBladeWithSign](1 << (basis.signature.vectorsCount * 2))

  private def getPos(left: BasisBlade, right: BasisBlade): Int =
    (left.bits << basis.signature.vectorsCount) + right.bits

  override def apply(left: BasisBlade, right: BasisBlade): BasisBladeWithSign =
    val pos = getPos(left, right)
    val result = data(pos)
    if (result != null) {
      result
    } else {
      val value = rule(left, right)
      data(pos) = value
      value
    }

object CachedMultiplication:
  def apply(f: (BasisBlade, BasisBlade) => BasisBladeWithSign)(using basis: Basis): CachedMultiplication =
    val rule: Multiplication = (left, right) => f(left, right)
    new CachedMultiplication(rule, basis)
