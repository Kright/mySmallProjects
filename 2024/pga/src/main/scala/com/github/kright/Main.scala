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

    //    val a = MultiVector[Symbolic](
    //      "x" -> Symbolic.Symbol("ax"),
    //      "y" -> Symbolic.Symbol("ay"),
    //      "z" -> Symbolic.Symbol("az"),
    //      "w" -> Symbolic.Constant(1.0),
    //    )
    //
    //    val b = MultiVector[Symbolic](
    //      "x" -> Symbolic.Symbol("bx"),
    //      "y" -> Symbolic.Symbol("by"),
    //      "w" -> Symbolic.Constant(1.0),
    //    )

    //    println(a.wedge(b).toMultilineString)

    // line
    println(point(1, 0, 0).wedge(point(0, 1, 0)).toMultilineString)

    // plane
    val xy0 = point(0, 0, 0).wedge(point(2, 0, 0)).wedge(point(0, 2, 0)).normalizedByWeight
    println(s"xy0 = ${xy0}")

    val xy_up1 = point(0, 0, 1).wedge(point(2, 0, 1)).wedge(point(0, 2, 1)).normalizedByWeight
    println(s"xy_up1 = ${xy_up1}")

    val p = point(2, 7, 10)

    val mirrored = xy0.geometricAntiproduct(p).geometricAntiproduct(xy0.antiReverse)
    println(mirrored / mirrored.weight.norm)

    val mirroredP = xy_up1 ⟇ xy0 ⟇ p ⟇ xy0.antiReverse ⟇ xy_up1.antiReverse
    println(s"mirroredP = ${mirroredP.filter((b, v) => v != 0.0)}")

    val moveUp2 = xy_up1 ⟇ xy0

    println(s"move up by 2 = ${moveUp2}")

    val plane45 = point(0, 0, 0).wedge(point(1, 0, 0)).wedge(point(0, 1, 1)).normalizedByWeight
    println(s"plane45 = ${plane45}")

    println(s"rot90 = ${plane45 ⟇ xy0}")
    println(s"rot90 up1 = ${plane45 ⟇ xy_up1}")
  }


def printMultiplicationTables()(using Basis): Unit = {
  val maxLen = 5

  println("geometric")
  println(basis.geometric.toPrettyString(basis.bladesByOrder, maxLen))
  println("geometricAntiproduct")
  println(basis.geometricAntiproduct.toPrettyString(basis.bladesByOrder, maxLen))

  println("dot")
  println(basis.dot.toPrettyString(basis.bladesByOrder, maxLen))
  println("antidot")
  println(basis.dotAntiproduct.toPrettyString(basis.bladesByOrder, maxLen))

  println("wedge")
  println(basis.wedge.toPrettyString(basis.bladesByOrder, maxLen))
  println("antiwedge")
  println(basis.wedgeAntiproduct.toPrettyString(basis.bladesByOrder, maxLen))
}
