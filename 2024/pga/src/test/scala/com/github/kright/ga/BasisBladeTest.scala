package com.github.kright.ga

import com.github.kright.ga.Generators.forAnyBasis
import org.scalatest.funsuite.AnyFunSuite

class BasisBladeTest extends AnyFunSuite:
  test("any complement") {
    forAnyBasis {
      for (blade <- basis.bladesByOrder) {
        val anyComplement = blade.anyComplement
        assert(basis.rules.geometric(blade, anyComplement).basisBlade == basis.antiScalarBlade, s"${blade} geometric ${anyComplement} = ${basis.rules.geometric(blade, anyComplement)}")
      }
    }
  }

  test("right and left complement") {
    forAnyBasis {
      val rule = MultiplicationRules()
      for (blade <- basis.bladesByOrder) {
        val bladeWithSign = BasisBladeWithSign(blade)
        assert(basis.rules.geometric(bladeWithSign, rule.rightComplement(blade)) == BasisBladeWithSign(basis.antiScalarBlade, Sign.Positive))
        assert(basis.rules.geometric(rule.leftComplement(blade), bladeWithSign) == BasisBladeWithSign(basis.antiScalarBlade, Sign.Positive))
      }
    }
  }
