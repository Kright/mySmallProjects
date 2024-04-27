package com.github.kright.ga

import scala.language.implicitConversions
import scala.util.chaining.*

class SingleOpTable(private val basis: Basis) extends SingleOp:
  private val data = new Array[BasisBladeWithSign](basis.signature.bladesCount)

  private def getPos(x: BasisBlade): Int =
    x.bits

  private def update(x: BasisBlade, value: BasisBladeWithSign): Unit = {
    data(getPos(x)) = value
  }

  override def apply(x: BasisBlade): BasisBladeWithSign =
    data(getPos(x))

  private def toPrettyString(v: BasisBladeWithSign): String =
    v.sign match
      case Sign.Positive => v.basisBlade.toString
      case Sign.Negative => s"-${v.basisBlade}"
      case Sign.Zero => "0"

  def toPrettyString(bladesOrder: IndexedSeq[BasisBlade]): String =
    bladesOrder.map(b => s"${b} -> ${toPrettyString(this(b))}").mkString("[", ", ", "]")


object SingleOpTable:
  def apply(op: SingleOp)(using basis: Basis): SingleOpTable =
    new SingleOpTable(basis).tap { table =>
      for (v <- basis.blades) {
        table(v) = op(v)
      }
    }
