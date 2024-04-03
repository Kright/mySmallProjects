package com.github.kright.symbolic

class SymbolicTransformSeq(val rules: Seq[SymbolicPartialTransform]) extends SymbolicPartialTransform:
  override def apply(symbolic: Symbolic): Option[Symbolic] =
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
