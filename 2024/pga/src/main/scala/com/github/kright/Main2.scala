package com.github.kright

import com.github.kright.ga.BasisPGA2.*
import com.github.kright.ga.{Basis, MultiVector}
import com.github.kright.symbolic.SymbolicStr
import com.github.kright.symbolic.SymbolicStr.given


@main
def main2(): Unit = Basis.pga2.use {
  //  val rot = rotate(math.sqrt(3) / 2, 0.5).normalizedByWeight
  val rot = rotate(0.999, math.sqrt(1.0 - 0.999 * 0.999)).normalizedByWeight
  val tr = translate(1.0, 2.0).normalizedByWeight
  
  val shifted = tr.mapValues(SymbolicStr(_)).geometricAntiproductSandwich(point(SymbolicStr("x"), SymbolicStr("y")))
  println(shifted.toPrettyMultilineString)

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

  println(point(0.0, 0.0).geometric(vector(1.0, 0.0)).withoutZeros)
  println(point(0.0, 1.0).geometric(vector(1.0, 0.0)).withoutZeros)
  println(point(1.0, 0.0).geometric(vector(1.0, 0.0)).withoutZeros)
  println(point(1.0, 1.0).geometric(vector(1.0, 0.0)).withoutZeros)
  println()

  println(tr.geometricAntiproductSandwich(point(1.0, 0.0)).withoutZeros)
  println(tr.geometricAntiproductSandwich(vector(1.0, 0.0)).withoutZeros)
}

@main
def mainTranslate() = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*

  val motionOp = makeTranslate(1.0, 2.0, 4.0)

  println(motionOp.mapValues(SymbolicStr(_)).geometricAntiproductSandwich(point(SymbolicStr("x"), SymbolicStr("y"), SymbolicStr("z"))).toPrettyMultilineString)
  println(motionOp.geometricAntiproductSandwich(point(4.0, 8.0, 9.0)).withoutZeros)

  val motionOpS = motionOp.mapValues(SymbolicStr(_))

  val f = vector[Double | String]("fx", "fy", "fz").mapValues(SymbolicStr(_))
  val r = point[Double | String]("rx", "ry", "rz", 1.0).mapValues(SymbolicStr(_))
  //  val moment = r.wedge(f)
  val moment = f.wedge(r)

  val shiftedF = motionOpS.geometricAntiproductSandwich(f).withoutZeros
  val shiftedR = motionOpS.geometricAntiproductSandwich(r).withoutZeros
  val expectedMovedMoment = shiftedF.wedge(shiftedR)

  println(s"shiftedF = ${shiftedF.toPrettyMultilineString}")
  println(s"shiftedR = ${shiftedR.toPrettyMultilineString}")

  val movedMoment = motionOpS.geometricAntiproductSandwich(moment)

  println(
    s"""
       |moment = ${moment.toPrettyMultilineString}
       |movedMoment = ${movedMoment.toPrettyMultilineString}
       |expected = ${expectedMovedMoment.toPrettyMultilineString}
       |""".stripMargin)
}

// todo rename geometricAntiProduct to antiGeometric and etc
