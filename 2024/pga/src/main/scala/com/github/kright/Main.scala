package com.github.kright.ga

type M = MultiVector[Double]

@main
def main(): Unit =
  Basis.pga3.use {
    def point(x: Double, y: Double, z: Double): M =
      MultiVector[Double](
        "x" -> x,
        "y" -> y,
        "z" -> z,
        "w" -> 1,
      )

    def vector(x: Double, y: Double, z: Double): M =
      MultiVector[Double](
        "x" -> x,
        "y" -> y,
        "z" -> z,
      )

    val a = MultiVector[Symbolic](
      "x" -> Symbolic.Symbol("ax"),
      "y" -> Symbolic.Symbol("ay"),
      "z" -> Symbolic.Symbol("az"),
      "w" -> Symbolic.Constant(1.0),
    )

    val b = MultiVector[Symbolic](
      "x" -> Symbolic.Symbol("bx"),
      "y" -> Symbolic.Symbol("by"),
      "w" -> Symbolic.Constant(1.0),
    )

    println(a.wedge(b).toMultilineString)

    // line
    println(point(1, 0, 0).wedge(point(0, 1, 0)).toMultilineString)

    // plane
    println(point(1, 0, 0).wedge(point(0, 1, 0)).wedge(point(1, 1, 0)))
  }


def printMultiplicationTables()(using Basis): Unit = {
  val maxLen = 5

  println("geometric")
  println(basis.geometric.toPrettyString(summon[Basis].bladesByOrder, maxLen))
  println("geometricAntiproduct")
  println(basis.geometricAntiproduct.toPrettyString(summon[Basis].bladesByOrder, maxLen))

  println("dot")
  println(basis.dot.toPrettyString(summon[Basis].bladesByOrder, maxLen))
  println("antidot")
  println(basis.dotAntiproduct.toPrettyString(summon[Basis].bladesByOrder, maxLen))

  println("wedge")
  println(basis.wedge.toPrettyString(summon[Basis].bladesByOrder, maxLen))
  println("antiwedge")
  println(basis.wedgeAntiproduct.toPrettyString(summon[Basis].bladesByOrder, maxLen))
}
