package com.github.kright.symbolic


sealed trait Symbolic[+F, +S]

object Symbolic:
  case class Symbol[+A](a: A) extends Symbolic[Nothing, A]:
    override def toString: String = a.toString

  case class Func[+F, +A](func: F, args: Seq[Symbolic[F, A]]) extends Symbolic[F, A]

  def apply[F, S](f: F, a: Seq[Symbolic[F, S]]): Func[F, S] = Symbolic.Func[F, S](f, a)

  def apply[F, S](a: S): Symbolic[F, S] = Symbolic.Symbol(a)

  extension [F, S](expr: Symbolic[F, S])
    def isFunc: Boolean =
      expr match
        case _: Symbolic.Func[F, S] => true
        case _ => false

    def isSymbol: Boolean =
      expr match
        case _: Symbolic.Symbol[S] => true
        case _ => false

    def mapSymbols[S2](f: S => S2): Symbolic[F, S2] =
      expr match
        case Symbol(symbol) => Symbol(f(symbol))
        case Func(func, args) => Func(func, args.map(_.mapSymbols(f)))

    def flatMapSymbols[S2](f: S => Symbolic[F, S2]): Symbolic[F, S2] =
      expr match
        case Symbol(symbol) => f(symbol)
        case Func(func, args) => Func(func, args.map(_.flatMapSymbols(f)))

    def mapFunctions[F2](f: F => F2): Symbolic[F2, S] =
      expr match
        case sa@Symbol(_) => sa
        case Func(func, args) => Func(f(func), args.map(_.mapFunctions(f)))

    def flatMapFunctions[F2](f: (F, Seq[Symbolic[F2, S]]) => Symbolic[F2, S]): Symbolic[F2, S] =
      expr match
        case sa@Symbol(_) => sa
        case Func(func, args) => f(func, args.map(_.flatMapFunctions(f)))

    def map[F2, S2](fSymbol: S => S2, fFunc: F => F2): Symbolic[F2, S2] =
      expr match
        case Symbol(a) => Symbol(fSymbol(a))
        case Func(func, args) => Func(fFunc(func), args.map(_.map(fSymbol, fFunc)))

    def flatMap[F2, S2](fmapSymbol: S => Symbolic[F2, S2],
                        fmapFunc: (F, Seq[Symbolic[F2, S2]]) => Symbolic[F2, S2]): Symbolic[F2, S2] =
      expr match
        case Symbol(a) => fmapSymbol(a)
        case Func(func, args) => fmapFunc(func, args.map(_.flatMap(fmapSymbol, fmapFunc)))

  extension (expr: SimpleSymbolic)
    def isZero: Boolean =
      expr match
        case Symbolic.Symbol(0.0) => true
        case _ => false

    def isOne: Boolean =
      expr match
        case Symbolic.Symbol(1.0) => true
        case _ => false

    def isFunc(name: String): Boolean =
      expr match
        case Symbolic.Func(realName, _) if name == realName => true
        case _ => false

  implicit val symbolicNumeric: Numeric[SimpleSymbolic] = new Numeric[SimpleSymbolic] {
    override def plus(x: SimpleSymbolic, y: SimpleSymbolic): SimpleSymbolic = SimpleSymbolic("+", Seq(x, y))

    override def minus(x: SimpleSymbolic, y: SimpleSymbolic): SimpleSymbolic = SimpleSymbolic("-", Seq(x, y))

    override def times(x: SimpleSymbolic, y: SimpleSymbolic): SimpleSymbolic = SimpleSymbolic("*", Seq(x, y))

    override def negate(x: SimpleSymbolic): SimpleSymbolic = SimpleSymbolic("-", Seq(x))

    override def fromInt(x: Int): SimpleSymbolic = SimpleSymbolic(x.toDouble)

    override def parseString(str: String): Option[SimpleSymbolic] = ???

    override def toInt(x: SimpleSymbolic): Int = ???

    override def toLong(x: SimpleSymbolic): Long = ???

    override def toFloat(x: SimpleSymbolic): Float = ???

    override def toDouble(x: SimpleSymbolic): Double = ???

    override def compare(x: SimpleSymbolic, y: SimpleSymbolic): Int = ???
  }
