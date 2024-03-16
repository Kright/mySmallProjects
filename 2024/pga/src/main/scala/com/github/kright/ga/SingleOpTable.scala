package com.github.kright.ga

import scala.language.implicitConversions
import scala.util.chaining.*

class SingleOpTable(private val basis: Basis) extends SingleOp:
  private val data = new Array[BasisBladeWithSign](basis.bladesCount)

  private def getPos(x: BasisBlade): Int =
    x.bits

  private def update(x: BasisBlade, value: BasisBladeWithSign): Unit = {
    data(getPos(x)) = value
  }

  override def apply(x: BasisBlade): BasisBladeWithSign =
    data(getPos(x))


object SingleOpTable:
  def apply(op: SingleOp)(using basis: Basis): SingleOpTable =
    new SingleOpTable(basis).tap { table =>
      for (v <- basis.blades) {
        table(v) = op(v)
      }
    }
