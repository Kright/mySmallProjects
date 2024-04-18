package com.github.kright


import com.github.kright.ga.{Basis, MultiVector}
import com.github.kright.symbolic.SymbolicStr
import com.github.kright.symbolic.SymbolicStr.given

import scala.math.Numeric.Implicits.infixNumericOps


@main
def translate2d(): Unit = Basis.pga2.use {
  import com.github.kright.ga.BasisPGA2.*
  println(rotate(SymbolicStr("cos"), SymbolicStr("sin")).toPrettyMultilineString)
  val tr = translateX2(SymbolicStr("dx") * SymbolicStr(0.5), SymbolicStr("dy") * SymbolicStr(0.5))
  println("antireverse =" + tr.antiReverse.toPrettyMultilineString)
  println(tr.antiSandwich(point(SymbolicStr("x"), SymbolicStr("y"))).toPrettyMultilineString)
  println(tr.antiSandwich(vector(SymbolicStr("x"), SymbolicStr("y"))).toPrettyMultilineString)
}

@main
def mirror3d(): Unit = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*
  val mirror = plane(1.0, 0.0, 0.0, 0.0)

  println(
    s"""
       |mirror = ${plane(1.0, 2.0, 3.0, 0.0)}
       |shiftedMirror = ${plane(1.0, 2.0, 3.0, math.sqrt(1 + 4 + 9))}
       |manually shiftedMirror = ${translate(1.0, 2.0, 3.0).antiSandwich(plane(1.0, 2.0, 3.0, 0.0))}
       |""".stripMargin)

  println(mirror.mapValues(SymbolicStr(_)).antiSandwich(point(SymbolicStr("x"), SymbolicStr("y"), SymbolicStr("z"))).toPrettyMultilineString)
  println(mirror.mapValues(SymbolicStr(_)).antiSandwich(vector(SymbolicStr("x"), SymbolicStr("y"), SymbolicStr("z"))).toPrettyMultilineString)
}

@main
def invalidPlane(): Unit = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*

  val p = point(0.0, 0.0, 0.0) wedge point(1.0, 0.0, 0.0) wedge point(2.0, 0.0, 0.0)
  println(p.toMultilineString)
}

@main
def validPlane3d(): Unit = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*

  println((point(0.0, 0.0, 0.0) ∧ point(1.0, 0.0, 0.0) ∧ point(0.0, 1.0, 0.0)).withoutZeros.toMultilineString)
  println((point(0.0, 0.0, 0.0) ∧ point(1.0, 0.0, 0.0) ∧ point(0.0, 2.0, 0.0)).withoutZeros.toMultilineString)
}

@main
def translate3d() = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*

  val translateXYZ = translateX2(SymbolicStr("dx") * SymbolicStr(0.5), SymbolicStr("dy") * SymbolicStr(0.5), SymbolicStr("dz") * SymbolicStr(0.5))

  println("antireverse =" + translateXYZ.antiReverse.toPrettyMultilineString)

  val f = vector[Double | String]("fx", "fy", "fz").mapValues(SymbolicStr(_))
  val r = point[Double | String]("rx", "ry", "rz", 1.0).mapValues(SymbolicStr(_))
  val moment = f.wedge(r)

  val shiftedF = translateXYZ.antiSandwich(f).withoutZeros
  val shiftedR = translateXYZ.antiSandwich(r).withoutZeros
  val expectedMovedMoment = shiftedF.wedge(shiftedR)

  println(s"shiftedF = ${shiftedF.toPrettyMultilineString}")
  println(s"shiftedR = ${shiftedR.toPrettyMultilineString}")

  val movedMoment = translateXYZ.antiSandwich(moment)

  println(
    s"""
       |moment = ${moment.toPrettyMultilineString}
       |movedMoment = ${movedMoment.toPrettyOrderedMultilineString}
       |expected = ${expectedMovedMoment.toPrettyOrderedMultilineString}
       |""".stripMargin)
}

// todo rename geometricAntiProduct to antiGeometric and etc
