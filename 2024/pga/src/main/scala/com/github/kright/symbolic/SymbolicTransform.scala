package com.github.kright.symbolic

trait SymbolicTransform[F, S] extends (Symbolic[F, S] => Symbolic[F, S])
