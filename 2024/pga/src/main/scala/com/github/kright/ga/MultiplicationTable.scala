package com.github.kright.ga

import scala.language.implicitConversions
import scala.util.chaining.*

class MultiplicationTable(private val basis: Basis) extends Multiplication:
  private val data = new Array[BasisBladeWithSign](1 << (basis.vectorsCount * 2))

  private def getPos(left: BasisBlade, right: BasisBlade): Int =
    (left.bits << basis.vectorsCount) + right.bits

  private def update(left: BasisBlade, right: BasisBlade, value: BasisBladeWithSign): Unit = {
    data(getPos(left, right)) = value
  }

  override def apply(left: BasisBlade, right: BasisBlade): BasisBladeWithSign =
    data(getPos(left, right))

  private def toPrettyString(v: BasisBladeWithSign): String =
    v.sign match
      case Sign.Positive => v.basisBlade.toString
      case Sign.Negative => s"-${v.basisBlade}"
      case Sign.Zero => "0"

  def toPrettyString(bladesOrder: IndexedSeq[BasisBlade]): String =
    val strings: IndexedSeq[IndexedSeq[String]] =
      bladesOrder.map { left =>
        bladesOrder.map { right =>
          toPrettyString(this(left, right))
        }
      }

    val maxLen = strings.view.flatten.map(_.length).max
    val padded = strings.map(_.map(s => s" ".repeat(maxLen - s.length) + s))
    padded.map(_.mkString("|")).mkString("\n")


object MultiplicationTable:
  def apply(op: (BasisBlade, BasisBlade) => BasisBladeWithSign)(using basis: Basis): MultiplicationTable =
    new MultiplicationTable(basis).tap { table =>
      for (left <- basis.blades; right <- basis.blades) {
        table(left, right) = op(left, right)
      }
    }

  def fromRule(op: MultiplicationRule => (BasisBlade, BasisBlade) => BasisBladeWithSign)(using basis: Basis): MultiplicationTable =
    val rule = MultiplicationRule()
    val func = op(rule)
    apply(func)

