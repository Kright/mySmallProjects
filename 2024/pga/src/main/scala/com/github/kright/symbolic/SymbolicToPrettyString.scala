package com.github.kright.symbolic

class SymbolicToPrettyString extends SymbolicRecursiveTransformHelper[SimpleSymbolic.Func, SimpleSymbolic.Symbol]:
  override protected def patternTransform(symbolic: SimpleSymbolic): Option[SimpleSymbolic] =
    Option {
      symbolic match
        case Symbolic.Symbol(arg) => arg match
          case v: Double => SimpleSymbolic(v.toString)
          case s: String => SimpleSymbolic(s)
        case Symbolic.Func("*", elems) => SimpleSymbolic(elems.mkString("", " * ", ""))
        case Symbolic.Func("+", elems) => SimpleSymbolic(elems.map(_.toString).mkString("(", " + ", ")").replace("+ -", "- "))
        case Symbolic.Func("-", elems) if elems.size == 1 => SimpleSymbolic(s"-${elems.head}")
        case Symbolic.Func(name, elems) if elems.size == 2 => SimpleSymbolic(elems.mkString("(", s" $name ", ")"))
        case Symbolic.Func(name, elems) => SimpleSymbolic(elems.mkString(s"${name}(", ", ", ")"))
    }
