package com.github.kright.symbolic

class SymbolicTransformAny[F, S](val rules: Seq[SymbolicPartialTransform[F, S]]) extends SymbolicPartialTransform[F, S]:
  override def apply(symbolic: Symbolic[F, S]): Option[Symbolic[F, S]] =
    rules.view.flatMap(r => r(symbolic)).headOption