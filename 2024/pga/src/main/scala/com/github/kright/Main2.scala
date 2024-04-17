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

  val center = point[Double](0.0, 0.0, 0.0)
  val dx = vector[Double](1.0, 0.0, 0.0)
  val dy = vector[Double](0.0, 1.0, 0.0)
  val dz = vector[Double](0.0, 0.0, 1.0)

  val planeUp = center.wedge(center + dx).wedge(center + dy).withoutZeros
  println(planeUp.toMultilineString)

  val planeUpShifted = (center + dz).wedge(center + dx + dz).wedge(center + dy + dz).withoutZeros
  println(planeUpShifted.toMultilineString)

  val planeUpShiftedV2 = (center + dz * 2).wedge(center + dx + dz * 2).wedge(center + dy + dz * 2).withoutZeros
  println(planeUpShiftedV2.toMultilineString)

  val motion = planeUp.geometricAntiproduct(planeUpShifted)
  val motionR = planeUp.geometricAntiproduct(planeUpShifted).reverse
  println(motion)
  println(motionR)

  println(motion.geometricAntiproductSandwich(center).withoutZeros.normalizedByWeight.toMultilineString)
  println(motionR.geometricAntiproductSandwich(center).withoutZeros.normalizedByWeight.toMultilineString)

  println("my translate")

  val motionOp = makeTranslate(1.0, 0.0, 0.0, shift = 1.0)


  println(motionOp)
  println(motionOp.geometricAntiproductSandwich(point(0.0, 0.0, 0.0)).withoutZeros)

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
  //
  //  val simplifier = SymbolicSimplifier()
  //
  //  println(r.wedge(f).toPrettyMultilineString)
}
