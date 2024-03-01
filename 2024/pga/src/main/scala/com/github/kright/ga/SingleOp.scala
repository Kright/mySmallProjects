package com.github.kright.ga

trait SingleOp:
  def apply(x: BasisBlade): BasisBladeWithSign

  def apply(x: BasisBladeWithSign): BasisBladeWithSign =
    apply(x.basisBlade) * x.sign
