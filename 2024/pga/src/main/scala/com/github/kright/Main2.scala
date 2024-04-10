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
    point[SimpleSymbolic](SimpleSymbolic("fx"), SimpleSymbolic("fy"), SimpleSymbolic.one)
      .wedge(point(SimpleSymbolic("rx"), SimpleSymbolic("ry"), SimpleSymbolic.one)).toPrettyMultilineString)


  //  println(
  //    vector[Symbolic](SimpleSymbolic("fx"), SimpleSymbolic("fy"))
  //      .wedge(point(SimpleSymbolic("rx"), SimpleSymbolic("ry"), Constant(1.0))).toMultilineString)
  //
  //
  //
  //  println(
  //    vector[Symbolic](SimpleSymbolic("fx"), SimpleSymbolic("fy"))
  //      .geometricAntiproduct(point(SimpleSymbolic("rx"), SimpleSymbolic("ry"), Constant(1.0))).toMultilineString)
}

@main
def d3() = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*

  val f = point[Double | String]("fx", "fy", "fz", 1.0).mapValues(SimpleSymbolic(_))
  val r = point[Double | String]("rx", "ry", "rz", 1.0).mapValues(SimpleSymbolic(_))
  
  println(f.wedge(r).toPrettyMultilineString)
}
