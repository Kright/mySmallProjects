package com.github.kright.symbolic

abstract class SymbolicRecursiveTransformHelper extends SymbolicPartialTransform:
  protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic]

  final override def apply(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
    symbolic match
      case s: Symbolic.Symbol[SimpleSymbolic.Symbol] => patternTransform(s)
      case f: Symbolic.Func[SimpleSymbolic.Func, SimpleSymbolic.Symbol] => {
        apply(f.args) match
          case None => patternTransform(f)
          case Some(newElems) => Option {
            val newF = Symbolic.Func(f.func, newElems)
            patternTransform(newF).getOrElse(newF)
          }
      }
