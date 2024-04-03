package com.github.kright.symbolic

import scala.annotation.tailrec

class SymbolicTransformRepeater(rule: SymbolicPartialTransform, maxRepeatCount: Int) extends SymbolicPartialTransform:
  require(maxRepeatCount > 0)

  override def apply(symbolic: Symbolic): Option[Symbolic] =
    rule(symbolic) match
      case None => None
      case Some(next) => Option(repeatRule(next, maxRepeatCount - 1))

  @tailrec
  private def repeatRule(current: Symbolic, remainingRetires: Int): Symbolic =
    if (remainingRetires > 0) {
      rule(current) match
        case None => current
        case Some(next) => repeatRule(next, remainingRetires - 1)
    }
    else current
    