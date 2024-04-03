package com.github.kright

import com.github.kright.ga.BasisPGA3.*
import com.github.kright.ga.{Basis, MultiVector, basis}
import com.github.kright.symbolic.*

type M = MultiVector[Double]

@main
def main(): Unit =
  Basis.pga3.use {
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

    val mass = MultiVector[Symbolic]("" -> Symbol("mass"))
    val velocity = MultiVector[Symbolic]("x" -> Symbol("vx"))
    val impulse: MultiVector[Symbolic] = mass ⟇ velocity
    println(s"mass = ${mass}")
    println(s"velocity = ${velocity}")
    println(s"impulse = ${mass.geometric(velocity).toPrettyMultilineString}")

    /*
      идея: скомбинировать в одной сущности импульс + момент импульса.
      как это может выглядеть: для 3д случая
     */

    println(
      point(Symbol("rx"), Symbol("ry"), Symbol("rz"), Constant(1.0))
        .geometric(vector(Symbol("px"), Symbol("py"), Symbol("pz"))).toPrettyMultilineString)

    {
      val r = point(Symbol("rx"), Symbol("ry"), Symbol("rz"), Constant(1.0))
      val f = vector[Symbolic](Symbol("fx"), Symbol("fy"), Symbol("fz"))
      println(r.wedge(f).toPrettyMultilineString)
      println(f.wedge(r).toPrettyMultilineString)
    }
  }

@main
def multiplyTrivectors(): Unit =
  Basis.pga3.use {
    def makeTrivector(number: Int) = MultiVector[Symbolic](
      "xyz" -> Symbol(s"xyz${number}"),
      "xyw" -> Symbol(s"xyw${number}"),
      "xwz" -> Symbol(s"xwz${number}"),
      "wyz" -> Symbol(s"wyz${number}"),
    )

    val trivector1 = makeTrivector(1)
    val trivector2 = makeTrivector(2)
    val trivector3 = makeTrivector(3)

    //    println(trivector1.toMultilineString)
    //    println(trivector1.geometric(trivector2).toMultilineString)
    //    println(trivector1.geometric(trivector2).geometric(trivector3).toMultilineString)

    //    val r = point(0.0, 0.0, 0.0) ∧ point(1.0, 0.0, 0.0) ∧ point(0.0, 1.0, 0.0)

    def makePoint(number: Int) = MultiVector[Symbolic](
      "x" -> Symbol(s"x${number}"),
      "y" -> Symbol(s"y${number}"),
      "z" -> Symbol(s"z${number}"),
      "w" -> Symbol(s"w${number}"),
    )

    val r = makePoint(1) ⟑ makePoint(2) ⟑ makePoint(3)
    println(r.toPrettyMultilineString)
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

@main
def makeInertiaTensor(): Unit = Basis.pga3.use {
  import Symbolic._

  val t = point[Symbolic](Symbol("px"), Symbol("py"), Constant(0.0)).geometricAntiproduct(vector[Symbolic](Symbol("vx"), Symbol("vy"), Constant(0.0)))
  println(t.withoutZeros.toPrettyMultilineString)
}
