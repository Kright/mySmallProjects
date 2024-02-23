package com.github.kright.ga

import com.github.kright.ga.Generators.*
import org.scalacheck.Gen
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks.forAll

class MultiplicationMultivectorTest extends AnyFunSuite:
  implicit val vectorEquality: Equality[MultiVector[Double]] = GaEquality.makeEquality(1e-9)

  def checkAssociativityForBasisBlades(makeOp: Basis => (MultiVector[Double], MultiVector[Double]) => MultiVector[Double]): Unit = {
    for (basis <- allBasisesSeq) {
      val op = makeOp(basis)

      forAll(basis.bladesGen(1), basis.bladesGen(1), basis.bladesGen(1)) { (a, b, c) =>
        val left = op(op(a, b), c)
        val right = op(a, op(b, c))
        assert(left === right)
      }
    }
  }

  def checkDistributivityForMultivectors(makeOp: Basis => (MultiVector[Double], MultiVector[Double]) => MultiVector[Double]): Unit = {
    for (basis <- allBasisesSeq) {
      val op = makeOp(basis)

      forAll(basis.multivectorsGen, basis.multivectorsGen, basis.multivectorsGen) { (a, b, c) =>
        assert(op(a + b, c) === op(a, c) + op(b, c))
        assert(op(c, a + b) === op(c, a) + op(c, b))
      }
    }
  }

  def checkAssociativityForMultivectors(makeOp: Basis => (MultiVector[Double], MultiVector[Double]) => MultiVector[Double]): Unit = {
    for (basis <- allBasisesSeq) {
      val op = makeOp(basis)

      forAll(basis.multivectorsGen, basis.multivectorsGen, basis.multivectorsGen) { (a, b, c) =>
        assert(op(op(a, b), c) === op(a, op(b, c)))
      }
    }
  }

  test("geometric product distributivity for multivectors") {
    checkDistributivityForMultivectors { b => (l, r) => l.geometric(r) }
  }

  test("dot product distributivity for multivectors") {
    checkDistributivityForMultivectors { b => (l, r) => l.dot(r) }
  }

  test("wedge product distributivity for multivectors") {
    checkDistributivityForMultivectors { b => (l, r) => l.wedge(r) }
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


  test("wedge product with two same vectors is zero") {
    for (basis <- allBasisesSeq) {
      val zero = MultiVector.zero[Double](using basis)
      forAll(basis.bladesGen(1), basis.multivectorsGen) { (a, m) =>
        assert((a ∧ a) === zero)
        assert((a ∧ (m ∧ a)) === zero)
        assert(((a ∧ m) ∧ a) === zero)
      }
    }
  }

  test("geometric product preserves 1-blade length") {
    implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-9)

    for (basis <- allBasisesSeq.filter(_.zeros == 0)) {
      forAll(basis.bladesGen(1), basis.bladesGen(1), basis.bladesGen(1)) { (a, b, c) =>
        val ma = a.magnitude
        val mb = b.magnitude
        val mab = (a ⟑ b).magnitude
        assert(mab === ma * mb, s"wrong dist ${ma} * ${mb} = ${ma * mb} != ${mab}, basis = ${basis}")
      }
    }
  }

  test("geometric product preserves length for a product of blades") {
    implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-9)

    for (basis <- allBasisesSeq.filter(_.zeros == 0)) {
      forAll(Gen.containerOfN[Seq, MultiVector[Double]](4, basis.bladesGen(1))) { blades =>
        val mags = blades.map(_.magnitude)
        val totalMag = blades.reduce(_ ⟑ _).magnitude
        assert(mags.product === totalMag)
      }
    }
  }

  test("wedge product is antisymmetric") {
    for (basis <- allBasisesSeq) {
      basis.use {
        forAll(basis.bladesGen(1), basis.bladesGen(1)) { (a, b) =>
          assert(a ∧ b === -(b ∧ a))
        }
      }
    }
  }

  test("dot product is symmetric") {
    for (basis <- allBasisesSeq) {
      basis.use {
        forAll(basis.bladesGen(1), basis.bladesGen(1)) { (a, b) =>
          assert(a ⋅ b === b ⋅ a)
        }
      }
    }
  }

  test("geometric product as sum of wedge and dot") {
    for (basis <- allBasisesSeq) {
      basis.use {
        forAll(basis.multivectorsGen, basis.multivectorsGen) { (a, b) =>
          val w = a ∧ b
          val d = a ⋅ b
          val g = a ⟑ b

          assert(g === w + d,
            s"""g = ${g}
               |w = ${w}
               |d = ${d}
               |w + d = ${w + d}
               |""".stripMargin)
        }
      }
    }
  }
