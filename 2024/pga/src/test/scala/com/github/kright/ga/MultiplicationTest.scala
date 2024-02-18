package com.github.kright.ga

import com.github.kright.ga.Generators.*
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks.forAll

class MultiplicationTest extends AnyFunSuite:
  def checkAssociativityForBasisBlades(makeOp: Basis => (MultiVector[Double], MultiVector[Double]) => MultiVector[Double]): Unit = {
    for (basis <- allBasisesSeq) {
      val op = makeOp(basis)

      forAll(basis.basisMultivectorsGen, basis.basisMultivectorsGen, basis.basisMultivectorsGen) { (a, b, c) =>
        val left = op(op(a, b), c)
        val right = op(a, op(b, c))
      }
    }
  }

  def checkAssociativityForMultivectors(makeOp: Basis => (MultiVector[Double], MultiVector[Double]) => MultiVector[Double]): Unit = {
    for (basis <- allBasisesSeq) {
      val op = makeOp(basis)

      forAll(basis.multivectorsGen, basis.multivectorsGen, basis.multivectorsGen) { (a, b, c) =>
        val left = op(op(a, b), c)
        val right = op(a, op(b, c))
        assert(left.getSqrDist(right) < 1e-5, s"wrong dist ${left.getSqrDist(right)}, ${left}, ${right}")
      }
    }
  }

  test("geometric product associativity for blades") {
    checkAssociativityForBasisBlades { b => (l, r) => l.geometric(r) }
  }

  test("geometric product associativity for multivectors") {
    checkAssociativityForMultivectors { b => (l, r) => l.geometric(r) }
  }

  test("dot product associativity for blades") {
    checkAssociativityForBasisBlades { b => (l, r) => l.dot(r) }
  }

  test("wedge product associativity for blades") {
    checkAssociativityForBasisBlades { b => (l, r) => l.wedge(r) }
  }

  test("geometric product preserves 1-blade length") {
    implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-9)

    for (basis <- allBasisesSeq.filter(_.zeros == 0)) {
      forAll(basis.bladesGen(1), basis.bladesGen(1), basis.bladesGen(1)) { (a, b, c) =>
        val ma = a.magnitude
        val mb = b.magnitude
        val mc = c.magnitude
        val mab = (a ⟑ b).magnitude
        val mabc = (a ⟑ b ⟑ c).magnitude
        assert(mab === ma * mb, s"wrong dist ${ma} * ${mb} = ${ma * mb} != ${mab}, basis = ${basis}")
        assert(mabc === ma * mb * mc, s"wrong dist ${ma} * ${mb} * ${mc} = ${ma * mb * mc} != ${mabc}, basis = ${basis}")
      }
    }
  }