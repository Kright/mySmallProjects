package com.github.kright.ga

import scala.language.implicitConversions
import scala.util.chaining.*

class MultiplicationTable(private val basis: Basis):
  private val data = new Array[(BasisBlade, Sign)](1 << (basis.vectorsCount * 2))

  private def getPos(left: BasisBlade, right: BasisBlade): Int =
    (left.bits << basis.vectorsCount) + right.bits

  private def update(left: BasisBlade, right: BasisBlade, value: (BasisBlade, Sign)): Unit = {
    data(getPos(left, right)) = value
  }

  def apply(left: BasisBlade, right: BasisBlade): (BasisBlade, Sign) =
    data(getPos(left, right))

  private def toPrettyString(basisBlade: BasisBlade, sign: Sign): String =
    sign match
      case Sign.Positive => basisBlade.toString
      case Sign.Negative => s"-$basisBlade"
      case Sign.Zero => "0"

  def toPrettyString(bladesOrder: IndexedSeq[BasisBlade]): String =
    val strings: IndexedSeq[IndexedSeq[String]] =
      bladesOrder.map { left =>
        bladesOrder.map { right =>
          val (m, sign) = this (left, right)
          toPrettyString(m, sign)
        }
      }

    val maxLen = strings.view.flatten.map(_.length).max
    val padded = strings.map(_.map(s => s" ".repeat(maxLen - s.length) + s))
    padded.map(_.mkString("|")).mkString("\n")


object MultiplicationTable:
  def apply(op: (BasisBlade, BasisBlade) => (BasisBlade, Sign))(using basis: Basis): MultiplicationTable =
    new MultiplicationTable(basis).tap { table =>
      for (left <- basis.blades; right <- basis.blades) {
        table(left, right) = op(left, right)
      }
    }

  def geometric(using basis: Basis): MultiplicationTable =
    val rule = MultiplicationRule()
    apply(rule.geometric)

  def dot(using basis: Basis): MultiplicationTable =
    val rule = MultiplicationRule()
    apply(rule.dot)

  def wedge(using basis: Basis): MultiplicationTable =
    val rule = MultiplicationRule()
    apply(rule.wedge)
