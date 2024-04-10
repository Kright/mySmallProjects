package com.github.kright.symbolic

import com.github.kright.symbolic.SimpleSymbolic.Func


class SymbolicToPrettyString extends SymbolicTransform[SimpleSymbolic.Func, SimpleSymbolic.Symbol, Nothing, String]:
  override def apply(expr: Symbolic[Func, SimpleSymbolic.Symbol]): Symbolic[Nothing, String] =
    expr.flatMap({
      case d: Double => Symbolic.Symbol[String](d.toString)
      case s: String => Symbolic.Symbol[String](s)
    }, {
      case ("*", elems) => Symbolic.Symbol[String](asSymbolValues(elems).mkString("", " * ", ""))
      case ("+", elems) => Symbolic.Symbol[String](asSymbolValues(elems).mkString("(", " + ", ")").replace("+ -", "- "))
      case ("-", elems) if elems.size == 1 => Symbolic.Symbol[String](s"-${asSymbol(elems.head).value}")
      case (name, elems) if elems.size == 2 => Symbolic.Symbol[String](asSymbolValues(elems).mkString("(", s" $name ", ")"))
      case (name, elems) => Symbolic.Symbol[String](asSymbolValues(elems).mkString(s"${name}(", ", ", ")"))
    })

  private def asSymbol(symbol: Symbolic[Nothing, String]): Symbolic.Symbol[String] =
    val result@Symbolic.Symbol(_) = symbol
    result

  private def asSymbolValues(symbols: Seq[Symbolic[Nothing, String]]) =
    symbols.map(asSymbol(_).value)