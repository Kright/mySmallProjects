package com.github.kright.symbolic

class SymbolicToPrettyString extends SymbolicRecursiveTransformHelper:
  override protected def patternTransform(symbolic: Symbolic): Option[Symbolic] =
    Option {
      symbolic match
        case c: Constant => Symbol(c.toString)
        case s: Symbol => s
        case Func("*", elems) => Symbol(elems.mkString("(", " * ", ")"))
        case Func("+", elems) => Symbol(elems.mkString("(", " + ", ")"))
        case Func("-", elems) if elems.size == 1 => Symbol(s"-${elems.head}")
        case Func(name, elems) if elems.size == 2 => Symbol(elems.mkString("(", s" $name ", ")"))
        case Func(name, elems) => Symbol(elems.mkString(s"${name}(", ", ", ")"))
    }

