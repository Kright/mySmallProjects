package com.github.kright.symbolic

class SymbolicTransformAny(val rules: Seq[SymbolicPartialTransform]) extends SymbolicPartialTransform:
  override def apply(symbolic: Symbolic): Option[Symbolic] =
    rules.view.flatMap(r => r(symbolic)).headOption