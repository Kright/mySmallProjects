package com.github.kright.ga

trait SingleOp:
  def apply(x: BasisBlade): BasisBladeWithSign

  def apply(x: BasisBladeWithSign): BasisBladeWithSign =
    apply(x.basisBlade) * x.sign

  private def toPrettyString(v: BasisBladeWithSign): String =
    v.sign match
      case Sign.Positive => v.basisBlade.toString
      case Sign.Negative => s"-${v.basisBlade}"
      case Sign.Zero => "0"
  
  def toPrettyString(bladesOrder: IndexedSeq[BasisBlade]): String =
    bladesOrder.map(b => s"${b} -> ${toPrettyString(this(b))}").mkString("[", ", ", "]")
