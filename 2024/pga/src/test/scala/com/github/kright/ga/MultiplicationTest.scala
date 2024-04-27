package com.github.kright.ga

import com.github.kright.ga.Generators.forAnyBasis
import org.scalatest.funsuite.AnyFunSuite

class MultiplicationTest extends AnyFunSuite:

  test("dot is symmetric") {
    forAnyBasis {
      val dotTable = basis.rules.dot

      for (left <- basis.blades.filter(_.grade == 1);
           right <- basis.blades.filter(_.grade == 1) if left.bits <= right.bits) {
        assert(dotTable(left, right) == dotTable(right, left),
          s"""left = ${left},
             |right = ${right},
             |left dot right = ${dotTable(left, right)}
             |right dot left = ${dotTable(right, left)}
             |basis = ${basis}
             |""".stripMargin)
      }
    }
  }

  test("wedge is anti-symmetric") {
    forAnyBasis {
      val wedge = basis.rules.wedge

      for (left <- basis.blades.filter(_.grade == 1);
           right <- basis.blades.filter(_.grade == 1) if left.bits <= right.bits) {
        val w1 = wedge(left, right)
        val w2 = wedge(right, left)

        assert(w1.sign == -w2.sign && w1.basisBlade == w2.basisBlade)
      }
    }
  }

  test("geometric is a sum of dot and wedge") {
    forAnyBasis {
      for (left <- basis.blades.filter(_.grade == 1);
           right <- basis.blades.filter(_.grade == 1) if left.bits <= right.bits) {

        val w = basis.rules.wedge(left, right)
        val b = basis.rules.dot(left, right)
        val g = basis.rules.geometric(left, right)

        assert(g == w && b.sign == Sign.Zero || g == b && w.sign == Sign.Zero)
      }
    }
  }

  test("wedge grade is sums of grades or zero") {
    forAnyBasis {
      for (left <- basis.blades;
           right <- basis.blades) {

        val w = basis.rules.wedge(left, right)

        assert(w.basisBlade.grade == 0 || w.basisBlade.grade == left.grade + right.grade)
      }
    }
  }

  test("dot grade is sub of grades or zero") {
    forAnyBasis {
      for (left <- basis.blades;
           right <- basis.blades) {

        val w = basis.rules.dot(left, right)

        assert(w.basisBlade.grade == 0 || w.basisBlade.grade == math.abs(left.grade - right.grade))
      }
    }
  }
