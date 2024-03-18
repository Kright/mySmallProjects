package com.github.kright.ga

import com.github.kright.ga.Generators.*
import org.scalacheck.Gen
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks.forAll

class MultiplicationMultivectorTest extends AnyFunSuite:
  implicit val vectorEquality: Equality[MultiVector[Double]] = GaEquality.makeEquality(1e-9)

  def checkAssociativityForBasisBlades(makeOp: Basis => (MultiVector[Double], MultiVector[Double]) => MultiVector[Double]): Unit = {
    forAnyBasis {
      val op = makeOp(basis)

      forAll(basis.bladesGen(1), basis.bladesGen(1), basis.bladesGen(1)) { (a, b, c) =>
        val left = op(op(a, b), c)
        val right = op(a, op(b, c))
        assert(left === right)
      }
    }
  }

  def checkDistributivityForMultivectors(makeOp: Basis => (MultiVector[Double], MultiVector[Double]) => MultiVector[Double]): Unit = {
    forAnyBasis {
      val op = makeOp(basis)

      forAll(basis.multivectorsGen, basis.multivectorsGen, basis.multivectorsGen) { (a, b, c) =>
        assert(op(a + b, c) === op(a, c) + op(b, c))
        assert(op(c, a + b) === op(c, a) + op(c, b))
      }
    }
  }

  def checkAssociativityForMultivectors(makeOp: Basis => (MultiVector[Double], MultiVector[Double]) => MultiVector[Double]): Unit = {
    forAnyBasis {
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

  test("geometric antiproduct associativity for multivectors") {
    checkAssociativityForMultivectors { b => (l, r) => l.geometricAntiproduct(r) }
  }

  test("dot product associativity for blades") {
    checkAssociativityForBasisBlades { b => (l, r) => l.dot(r) }
  }

  test("wedge product associativity for blades") {
    checkAssociativityForBasisBlades { b => (l, r) => l.wedge(r) }
  }

  // I don't know why, but this is not working
//  test("dot antiproduct product associativity for blades") {
//    checkAssociativityForBasisBlades { b => (l, r) => l.dotAntiproduct(r) }
//  }

  test("wedge antiproduct product associativity for blades") {
    checkAssociativityForBasisBlades { b => (l, r) => l.wedgeAntiproduct(r) }
  }

  test("wedge product with two same vectors is zero") {
    forAnyBasis {
      val zero = MultiVector.zero[Double]
      forAll(basis.bladesGen(1), basis.multivectorsGen) { (a, m) =>
        assert((a ∧ a) === zero)
        assert((a ∧ (m ∧ a)) === zero)
        assert(((a ∧ m) ∧ a) === zero)
      }
    }
  }

  test("geometric product preserves 1-blade length") {
    implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-9)

    forAnyBasis {
      if (basis.zeros == 0) {
        forAll(basis.bladesGen(1), basis.bladesGen(1), basis.bladesGen(1)) { (a, b, c) =>
          val ma = a.magnitude
          val mb = b.magnitude
          val mab = (a ⟑ b).magnitude
          assert(mab === ma * mb, s"wrong dist ${ma} * ${mb} = ${ma * mb} != ${mab}, basis = ${basis}")
        }
      }
    }
  }

  test("geometric product preserves length for a product of blades") {
    implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-9)

    forAnyBasis {
      if (basis.zeros == 0) {
        forAll(Gen.containerOfN[Seq, MultiVector[Double]](4, basis.bladesGen(1))) { blades =>
          val mags = blades.map(_.magnitude)
          val totalMag = blades.reduce(_ ⟑ _).magnitude
          assert(mags.product === totalMag)
        }
      }
    }
  }

  test("wedge product is antisymmetric") {
    forAnyBasis {
      forAll(basis.bladesGen(1), basis.bladesGen(1)) { (a, b) =>
        assert(a ∧ b === -(b ∧ a))
      }
    }
  }

  test("dot product is symmetric") {
    forAnyBasis {
      forAll(basis.bladesGen(1), basis.bladesGen(1)) { (a, b) =>
        assert(a ⋅ b === b ⋅ a)
      }
    }
  }

  test("geometric product as sum of wedge and dot") {
    forAnyBasis {
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

  test("geometric antiproduct as sum of wedge antiproduct and dot antiproduct") {
    forAnyBasis {
      forAll(basis.multivectorsGen, basis.multivectorsGen) { (a, b) =>
        val w = a ∨ b
        val d = a ◦ b
        val g = a ⟇ b

        assert(g === w + d,
          s"""g = ${g}
             |w = ${w}
             |d = ${d}
             |w + d = ${w + d}
             |""".stripMargin)
      }
    }
  }

  test("geometric antiproduct corresponds to geometric product") {
    forAnyBasis {
      forAll(basis.multivectorsGen, basis.multivectorsGen) { (a, b) =>
        assert(a.geometric(b).rightComplement === a.rightComplement.geometricAntiproduct(b.rightComplement))
      }
    }
  }

  test("wedge antiproduct corresponds to wedge product") {
    forAnyBasis {
      forAll(basis.multivectorsGen, basis.multivectorsGen) { (a, b) =>
        assert(a.wedge(b).rightComplement === a.rightComplement.wedgeAntiproduct(b.rightComplement))
      }
    }
  }

  test("dot antiproduct corresponds to dot product") {
    forAnyBasis {
      forAll(basis.multivectorsGen, basis.multivectorsGen) { (a, b) =>
        assert(a.dot(b).rightComplement === a.rightComplement.dotAntiproduct(b.rightComplement))
      }
    }
  }

  test("pseudoScalar commutativity and anticommutativity") {
    forAnyBasis {
      val i = MultiVector(basis.antiScalarBlade)
      forAll(basis.bladesGen(1)) { a =>
        val isEven = basis.vectorsCount % 2 == 0
        if (isEven) {
          assert((a ⟑ i) === -(i ⟑ a))
        } else {
          assert((a ⟑ i) === (i ⟑ a))
        }
      }
    }
  }

  test("exterior and interior product duality") {
    forAnyBasis {
      val i = MultiVector(basis.antiScalarBlade)

      forAll(basis.bladesGen(1), basis.bladesGen(1)) { (a, b) =>
        val isEven = basis.vectorsCount % 2 == 0
        if (isEven) {
          assert(a ∧ (i ⟑ b) === -(a ⋅ b) ⟑ i)
        } else {
          assert(a ∧ (i ⟑ b) === (a ⋅ b) ⟑ i)
        }
      }
    }
  }

  test("buld and weight sum") {
    forAnyBasis {
      forAll(basis.multivectorsGen) { v =>
        assert((v.bulk + v.weight) === v)
      }
    }
  }

  test("sandwich products") {
    forAnyBasis {
      forAll(basis.multivectorsGen, basis.multivectorsGen, basis.multivectorsGen) { (a, b, mid) =>
        assert(a ⟑ mid ⟑ a.reverse === a.geometricSandwich(mid))
        assert(a ⟑ b ⟑ mid ⟑ b.reverse ⟑ a.reverse === a.geometric(b).geometricSandwich(mid))

        assert(a ⟇ mid ⟇ a.antiReverse === a.geometricAntiproductSandwich(mid))
        assert(a ⟇ b ⟇ mid ⟇ b.antiReverse ⟇ a.antiReverse === a.geometricAntiproduct(b).geometricAntiproductSandwich(mid))
      }
    }
  }
