package com.github.kright.symbolic

import scala.annotation.tailrec

class SymbolicTransformRepeater[F, S](rule: SymbolicPartialTransform[F, S], maxRepeatCount: Int) extends SymbolicPartialTransform[F, S]:
  require(maxRepeatCount > 0)

  override def apply(symbolic: Symbolic[F, S]): Option[Symbolic[F, S]] =
    rule(symbolic) match
      case None => None
      case Some(next) => Option(repeatRule(next, maxRepeatCount - 1))

  @tailrec
  private def repeatRule(current: Symbolic[F, S], remainingRetires: Int): Symbolic[F, S] =
    if (remainingRetires > 0) {
      rule(current) match
        case None => current
        case Some(next) => repeatRule(next, remainingRetires - 1)
    }
    else current
    