package com.github.kright.ga

import org.scalatest.funsuite.AnyFunSuite

class MultiplicationRuleTest extends AnyFunSuite {
  private def allBasises = Seq(Basis.ga2, Basis.ga3, Basis.pga2, Basis.pga3)

  private val rulePga3 = Basis.pga3.use {
    MultiplicationRule()
  }

  test("dot for basis vectors") {
    for (basis <- allBasises) {
      basis.use {
        val rule = MultiplicationRule()
        for (v <- basis.vectors) {
          val vb = BasisBlade(v)
          assert(v.getSquareSign == rule.dot(vb, vb)._2)
        }
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

  test("dot wedge") {
    Basis.pga3.use {
      val a =
        MultiVector[Double](Map(
          BasisBlade("x") -> 1.0,
          BasisBlade("y") -> 10.0,
        ))

      val b = MultiVector[Double](Map(
        BasisBlade("y") -> 2.0,
        BasisBlade("xz") -> 100.0,
      ))

      println(rulePga3.wedge(a, b))
    }
  }
}
