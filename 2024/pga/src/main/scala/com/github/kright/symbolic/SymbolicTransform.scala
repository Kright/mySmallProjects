package com.github.kright.symbolic

trait SymbolicTransform[F, S, F2, S2] extends (Symbolic[F, S] => Symbolic[F2, S2])
