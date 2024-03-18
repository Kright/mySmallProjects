package com.github.kright.ga

sealed trait Sign:
  def toInt: Int

  def toDouble: Double = toInt.toDouble

  def *(other: Sign): Sign =
    Sign(this.toInt * other.toInt)

  def unary_- : Sign =
    Sign(-this.toInt)

  def power(pow: Int): Sign =
    require(pow >= 0)
    this match
      case Sign.Positive => Sign.Positive
      case Sign.Zero => if (pow == 0) Sign.Positive else Sign.Zero
      case Sign.Negative => if (pow % 2 == 1) Sign.Negative else Sign.Positive

object Sign:
  def apply(number: Int): Sign =
    if (number > 0) return Sign.Positive
    if (number < 0) return Sign.Negative
    Sign.Zero

  case object Zero extends Sign:
    override def toInt: Int = 0

  case object Positive extends Sign:
    override def toInt: Int = 1

  case object Negative extends Sign:
    override def toInt: Int = -1
