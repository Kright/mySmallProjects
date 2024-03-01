package com.github.kright.ga

trait Multiplication:
  def apply(left: BasisBlade, right: BasisBlade): BasisBladeWithSign

  def apply(left: BasisBladeWithSign, right: BasisBladeWithSign): BasisBladeWithSign =
    apply(left.basisBlade, right.basisBlade) * (left.sign * right.sign)
