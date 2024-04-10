package com.github.kright.symbolic

abstract class SymbolicRecursiveTransformHelper[F, S] extends SymbolicPartialTransform[F, S]:
  protected def patternTransform(symbolic: Symbolic[F, S]): Option[Symbolic[F, S]]

  final override def apply(symbolic: Symbolic[F, S]): Option[Symbolic[F, S]] =
    symbolic match
      case s@Symbolic.Symbol(_) => patternTransform(s)
      case f@Symbolic.Func(func, args) => {
        apply(args) match
          case None => patternTransform(f)
          case Some(newElems) => Option {
            val newF = Symbolic.Func(f.func, newElems)
            patternTransform(newF).getOrElse(newF)
          }
      }
