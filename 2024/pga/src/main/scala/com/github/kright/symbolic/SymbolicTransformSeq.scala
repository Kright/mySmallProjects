package com.github.kright.symbolic

class SymbolicTransformSeq[F, S](val rules: Seq[SymbolicPartialTransform[F, S]]) extends SymbolicPartialTransform[F, S]:
  override def apply(symbolic: Symbolic[F, S]): Option[Symbolic[F, S]] =
    var current = symbolic
    var anyUpdated = false
    for (r <- rules) {
      r(current) match
        case Some(next) => {
          current = next
          anyUpdated = true
        }
        case None =>
    }
    if (anyUpdated) Option(current) else None
