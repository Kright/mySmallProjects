package com.github.kright.symbolic

abstract class SymbolicRecursiveTransformHelper extends SymbolicPartialTransform:
  protected def patternTransform(symbolic: Symbolic): Option[Symbolic]

  final override def apply(symbolic: Symbolic): Option[Symbolic] =
    symbolic match
      case c: Constant => patternTransform(c)
      case s: Symbol => patternTransform(s)
      case f: Func => {
        apply(f.elems) match
          case None => patternTransform(f)
          case Some(newElems) => Option {
            val newF = Func(f.name, newElems)
            patternTransform(newF).getOrElse(newF)
          }
      }
