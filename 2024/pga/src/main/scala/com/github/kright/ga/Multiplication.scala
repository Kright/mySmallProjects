package com.github.kright.ga

trait Multiplication:
  def apply(left: BasisBlade, right: BasisBlade): BasisBladeWithSign

  def apply(left: BasisBladeWithSign, right: BasisBladeWithSign): BasisBladeWithSign =
    apply(left.basisBlade, right.basisBlade) * (left.sign * right.sign)

  private def toPrettyString(v: BasisBladeWithSign): String =
    v.sign match
      case Sign.Positive => v.basisBlade.toString
      case Sign.Negative => s"-${v.basisBlade}"
      case Sign.Zero => "0"

  def toPrettyString(bladesOrder: IndexedSeq[BasisBlade], setMaxLen: Int = 0): String =
    val strings: IndexedSeq[IndexedSeq[String]] =
      bladesOrder.map { left =>
        bladesOrder.map { right =>
          toPrettyString(this (left, right))
        }
      }

    val maxLen = math.max(setMaxLen, strings.view.flatten.map(_.length).max)
    val padded = strings.map(_.map(s => s" ".repeat(maxLen - s.length) + s))
    padded.map(_.mkString("|")).mkString("\n")