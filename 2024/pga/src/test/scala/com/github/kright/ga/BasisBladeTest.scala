package com.github.kright.ga

import org.scalatest.funsuite.AnyFunSuite

class BasisBladeTest extends AnyFunSuite {
  test("any complement") {
    for (basis <- Generators.allBasisesSeq) {
      basis.use {
        for (blade <- basis.bladesByOrder) {
          val anyComplement = blade.anyComplement
          assert(basis.geometric(blade, anyComplement).basisBlade == basis.antiScalarBlade, s"${blade} geometric ${anyComplement} = ${basis.geometric(blade, anyComplement)}")
        }
      }
    }
  }

  test("right and left complement") {
    for (basis <- Generators.allBasisesSeq) {
      basis.use {
        val rule = MultiplicationRule()
        for (blade <- basis.bladesByOrder) {
          val bladeWithSign = BasisBladeWithSign(blade)
          assert(basis.geometric(bladeWithSign, rule.rightComplement(blade)) == BasisBladeWithSign(basis.antiScalarBlade, Sign.Positive))
          assert(basis.geometric(rule.leftComplement(blade), bladeWithSign) == BasisBladeWithSign(basis.antiScalarBlade, Sign.Positive))
        }
      }
    }
  }
}
