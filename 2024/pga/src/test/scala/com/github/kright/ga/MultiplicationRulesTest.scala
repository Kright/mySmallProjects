package com.github.kright.ga

import com.github.kright.ga.Generators.forAnyBasis
import org.scalatest.funsuite.AnyFunSuite

class MultiplicationRulesTest extends AnyFunSuite:
  private val rulePga3 = Basis.pga3.use {
    MultiplicationRules()
  }

  test("dot for basis vectors") {
    forAnyBasis {
      val rule = MultiplicationRules()
      for (v <- basis.vectors) {
        val vb = BasisBlade(v)
        assert(v.getSquareSign == rule.dot(vb, vb)._2)
      }
    }

    assert(Basis.ga3.vectors.forall(_.getSquareSign == Sign.Positive))
    assert(Basis.ga2.vectors.forall(_.getSquareSign == Sign.Positive))

    Basis.pga3.use {
      assert(summon[Basis].vectors.map(BasisBlade(_)).map(b => rulePga3.dot(b, b)._2.toInt) == IndexedSeq(1, 1, 1, 0))
    }
  }

  test("dot for basis blades") {
    Basis.pga3.use {
      assert(Basis.pga3.bladesByOrder.map(b => rulePga3.dot(b, b)._2.toInt) == IndexedSeq(
        1,
        1, 1, 1, 0,
        -1, -1, -1, 0, 0, 0,
        -1, 0, 0, 0,
        0
      ))
    }
  }

  test("associativity") {
    forAnyBasis {
      val rule = MultiplicationRules()
      for (v <- basis.vectors) {
        val vb = BasisBlade(v)
        assert(v.getSquareSign == rule.dot(vb, vb)._2)
      }
    }

    assert(Basis.ga3.vectors.forall(_.getSquareSign == Sign.Positive))
    assert(Basis.ga2.vectors.forall(_.getSquareSign == Sign.Positive))

    Basis.pga3.use {
      assert(summon[Basis].vectors.map(BasisBlade(_)).map(b => rulePga3.dot(b, b)._2.toInt) == IndexedSeq(1, 1, 1, 0))
    }
  }

  // todo test that ga3 match with pga3 on same basis blades