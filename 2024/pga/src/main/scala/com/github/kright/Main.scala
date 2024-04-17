package com.github.kright

import com.github.kright.ga.BasisPGA3.*
import com.github.kright.ga.{Basis, MultiVector, basis}
import com.github.kright.symbolic.SymbolicStr
import com.github.kright.symbolic.SymbolicStr.{*, given}

type M = MultiVector[Double]

@main
def main(): Unit =
  Basis.pga3.use {
    //    val a = MultiVector[SymbolicStr](
    //      "x" -> SymbolicStr.Symbol("ax"),
    //      "y" -> SymbolicStr.Symbol("ay"),
    //      "z" -> SymbolicStr.Symbol("az"),
    //      "w" -> SymbolicStr.Constant(1.0),
    //    )
    //
    //    val b = MultiVector[SymbolicStr](
    //      "x" -> SymbolicStr.Symbol("bx"),
    //      "y" -> SymbolicStr.Symbol("by"),
    //      "w" -> SymbolicStr.Constant(1.0),
    //    )

    //    println(a.wedge(b).toMultilineString)

    // line
    println(point(1.0, 0.0, 0.0).wedge(point(0.0, 1.0, 0.0)).toMultilineString)

    // plane
    val xy0 = point(0.0, 0.0, 0.0).wedge(point(2.0, 0.0, 0.0)).wedge(point(0.0, 2.0, 0.0)).normalizedByWeight
    println(s"xy0 = ${xy0}")

    val xy_up1 = point(0.0, 0.0, 1.0).wedge(point(2.0, 0.0, 1.0)).wedge(point(0.0, 2.0, 1.0)).normalizedByWeight
    println(s"xy_up1 = ${xy_up1}")

    val p = point(2.0, 7.0, 10.0)

    val mirrored = xy0.geometricAntiproduct(p).geometricAntiproduct(xy0.antiReverse)
    println(mirrored / mirrored.weight.norm)

    val mirroredP = xy_up1 ⟇ xy0 ⟇ p ⟇ xy0.antiReverse ⟇ xy_up1.antiReverse
    println(s"mirroredP = ${mirroredP.withoutZeros}")

    val moveUp2 = xy_up1 ⟇ xy0

    println(s"move up by 2 = ${moveUp2}")

    val plane45 = point(0.0, 0.0, 0.0).wedge(point(1.0, 0.0, 0.0)).wedge(point(0.0, 1.0, 1.0)).normalizedByWeight
    println(s"plane45 = ${plane45}")

    val rot90 = plane45 ⟇ xy0
    println(s"rot90 = ${rot90.withoutZeros}")
    println(s"up2 ⟇ rot90 = ${(moveUp2 ⟇ rot90).withoutZeros}")
    println(s"rot90 ⟇ up2 = ${(rot90 ⟇ moveUp2).withoutZeros}")

    val randomPlane = point(0.0, 0.0, 0.0).wedge(point(1, 0.01, 0.01)).wedge(point(1.0, 2.0, 0.001)).normalizedByWeight

    println((randomPlane ⟇ xy0).normalizedByWeight.withoutZeros)

    val mass = MultiVector[SymbolicStr]("" -> SymbolicStr("mass"))
    val velocity = MultiVector[SymbolicStr]("x" -> SymbolicStr("vx"))
    val impulse: MultiVector[SymbolicStr] = mass ⟇ velocity
    println(s"mass = ${mass}")
    println(s"velocity = ${velocity}")
    println(s"impulse = ${mass.geometric(velocity).toPrettyMultilineString}")

    /*
      идея: скомбинировать в одной сущности импульс + момент импульса.
      как это может выглядеть: для 3д случая
     */

    println(
      point[SymbolicStr](SymbolicStr("rx"), SymbolicStr("ry"), SymbolicStr("rz"), SymbolicStr(1.0))
        .geometric(vector(SymbolicStr("px"), SymbolicStr("py"), SymbolicStr("pz"))).toPrettyMultilineString)

    {
      val r = point[SymbolicStr](SymbolicStr("rx"), SymbolicStr("ry"), SymbolicStr("rz"), SymbolicStr.one)
      val f = vector[SymbolicStr](SymbolicStr("fx"), SymbolicStr("fy"), SymbolicStr("fz"))
      println(r.wedge(f).toPrettyMultilineString)
      println(f.wedge(r).toPrettyMultilineString)
    }
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
