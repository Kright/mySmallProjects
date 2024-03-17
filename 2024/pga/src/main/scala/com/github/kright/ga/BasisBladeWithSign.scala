package com.github.kright.ga

case class BasisBladeWithSign(basisBlade: BasisBlade, sign: Sign = Sign.Positive):
  def *(anotherSign: Sign): BasisBladeWithSign = this.copy(sign = this.sign * anotherSign)
  
object BasisBladeWithSign:
  def zero(using basis: Basis): BasisBladeWithSign = BasisBladeWithSign(basis.scalarBlade, Sign.Zero)  
