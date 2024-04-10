package com.github.kright.symbolic

type SimpleSymbolic = Symbolic[String, Double | String]

object SimpleSymbolic:
  type Func = String
  type Symbol = Double | String

  def apply(func: String, args: Seq[SimpleSymbolic]): SimpleSymbolic =
    Symbolic.Func[String, Double | String](func, args)

  def apply(symbol: Double | String): SimpleSymbolic =
    Symbolic.Symbol(symbol)

  val zero: Symbolic.Symbol[Double | String] =
    Symbolic.Symbol(0.0)

  val one: Symbolic.Symbol[Double | String] =
    Symbolic.Symbol(1.0)
