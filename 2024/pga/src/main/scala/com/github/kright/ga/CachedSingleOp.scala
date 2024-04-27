package com.github.kright.ga

class CachedSingleOp(private val singleOp: SingleOp,
                     private val basis: Basis) extends SingleOp:
  private val data = new Array[BasisBladeWithSign](basis.signature.bladesCount)

  override def apply(x: BasisBlade): BasisBladeWithSign =
    val pos = x.bits
    val result = data(pos)
    if (result != null) {
      result
    } else {
      val value = singleOp(x)
      data(pos) = value
      value
    }

object CachedSingleOp:
  def apply(f: BasisBlade => BasisBladeWithSign)(using basis: Basis): CachedSingleOp =
    val singleOp: SingleOp = v => f(v)
    new CachedSingleOp(singleOp, basis)
