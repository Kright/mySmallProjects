package com.github.kright

import com.github.kright.ga.BasisPGA2.*
import com.github.kright.ga.{Basis, MultiVector}
import com.github.kright.symbolic.*

@main
def main2(): Unit = Basis.pga2.use {
  //  val rot = rotate(math.sqrt(3) / 2, 0.5).normalizedByWeight
  val rot = rotate(0.999, math.sqrt(1.0 - 0.999 * 0.999)).normalizedByWeight
  val tr = translate(1.0, 2.0).normalizedByWeight

  //  println(rot)
  //  println(tr)
  //  println(rot.geometricAntiproduct(tr))
  //  println(tr.geometricAntiproduct(rot))
  //
  //  println(rot.geometricAntiproductSandwich(point(0.0, 0.0)).withoutZeros)
  //  println(rot.geometricAntiproduct(tr).geometricAntiproductSandwich(point(0.0, 0.0)).withoutZeros)
  //  println(tr.geometricAntiproduct(rot).geometricAntiproductSandwich(point(0.0, 0.0)).withoutZeros)


  //  println(rot.geometricAntiproductSandwich(point(0.0, 1.0)).withoutZeros)
  //  println(rot.geometricAntiproductSandwich(point(1.0, 0.0)).withoutZeros)
  //  println(rot.geometricAntiproductSandwich(point(0.0, 0.0)).withoutZeros)

  println(point(0.0, 0.0).geometric(vector(1, 0)).withoutZeros)
  println(point(0.0, 1.0).geometric(vector(1, 0)).withoutZeros)
  println(point(1.0, 0.0).geometric(vector(1, 0)).withoutZeros)
  println(point(1.0, 1.0).geometric(vector(1, 0)).withoutZeros)
  println()

  println(tr.geometricAntiproductSandwich(point(1.0, 0.0)).withoutZeros)
  println(tr.geometricAntiproductSandwich(vector(1.0, 0.0)).withoutZeros)


  println()

  println(
    point[Symbolic](Symbol("fx"), Symbol("fy"), Constant(1.0))
      .wedge(point(Symbol("rx"), Symbol("ry"), Constant(1.0))).toPrettyMultilineString)


  //  println(
  //    vector[Symbolic](Symbol("fx"), Symbol("fy"))
  //      .wedge(point(Symbol("rx"), Symbol("ry"), Constant(1.0))).toMultilineString)
  //
  //
  //
  //  println(
  //    vector[Symbolic](Symbol("fx"), Symbol("fy"))
  //      .geometricAntiproduct(point(Symbol("rx"), Symbol("ry"), Constant(1.0))).toMultilineString)
}

@main
def d3() = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*

  println(
    point[Symbolic](Symbol("fx"), Symbol("fy"), Symbol("fz"), Constant(1.0))
      .wedge(point(Symbol("rx"), Symbol("ry"), Symbol("rz"), Constant(1.0))).toPrettyMultilineString)
}
