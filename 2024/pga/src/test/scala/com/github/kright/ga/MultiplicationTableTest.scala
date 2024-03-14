package com.github.kright.ga

import com.github.kright.ga.Generators.forAnyBasis
import org.scalatest.funsuite.AnyFunSuite

class MultiplicationTableTest extends AnyFunSuite:

  test("dot is symmetric") {
    forAnyBasis {
      val dotTable = basis.dot

      for (left <- basis.blades.filter(_.order == 1);
           right <- basis.blades.filter(_.order == 1) if left.bits <= right.bits) {
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
      val wedge = basis.wedge

      for (left <- basis.blades.filter(_.order == 1);
           right <- basis.blades.filter(_.order == 1) if left.bits <= right.bits) {
        val w1 = wedge(left, right)
        val w2 = wedge(right, left)

        assert(w1.sign == -w2.sign && w1.basisBlade == w2.basisBlade)
      }
    }
  }

  test("geometric is a sum of dot and wedge") {
    forAnyBasis {
      for (left <- basis.blades;
           right <- basis.blades if left.bits <= right.bits) {

        val w = basis.wedge(left, right)
        val b = basis.dot(left, right)
        val g = basis.geometric(left, right)

        assert(g == w && b.sign == Sign.Zero || g == b && w.sign == Sign.Zero)
      }
    }
  }
