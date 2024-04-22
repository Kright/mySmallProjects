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

  println(
    s"""
       |tr = ${tr.toPrettyMultilineString}
       |rightComplement = ${tr.rightComplement.toPrettyMultilineString}
       |leftComplement = ${tr.leftComplement.toPrettyMultilineString}
       |""".stripMargin)
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

  println(s"translateXYZ = ${translateXYZ.toPrettyMultilineString}")
  println(s"translateXYZ^2 = ${translateXYZ.antiGeometric(translateXYZ).toPrettyMultilineString}")
  println(s"translateXYZ.leftComplement = ${translateXYZ.leftComplement.toPrettyMultilineString}")
  println(s"translateXYZ.rightComplement = ${translateXYZ.rightComplement.toPrettyMultilineString}")
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
       |movedMoment = ${movedMoment.simplifiedOrdered.toPrettyMultilineString}
       |expected = ${expectedMovedMoment.simplifiedOrdered.toPrettyMultilineString}
       |""".stripMargin)
}


@main
def line2d() = Basis.pga2.use {
  import com.github.kright.ga.BasisPGA2.*

  val a = point(2.0, 0.0)
  val b = point(0.0, 2.0)
  val line = a wedge b


  println(line.toMultilineString)
  println(line.normalizedByWeight.toMultilineString)
}


@main
def line3d() = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*

  {
    val a = point(1.0, 0.0, 0.0)
    val b = point(0.0, 1.0, 0.0)
    val line = a wedge b

    println(line.toMultilineString)
    println(line.normalizedByWeight.toMultilineString)
  }

  {
    val a = point(SymbolicStr("ax"), SymbolicStr("ay"), SymbolicStr("az"))
    val b = point(SymbolicStr("bx"), SymbolicStr("by"), SymbolicStr("bz"))
    val line = a wedge b

    println(line.toPrettyMultilineString)
  }
}

@main
def plane2d() = Basis.pga2.use {
  import com.github.kright.ga.BasisPGA2.*

  val a = point(SymbolicStr("ax"), SymbolicStr("ay"))
  val b = point(SymbolicStr("bx"), SymbolicStr("by"))
  val plane = a wedge b

  println(plane.toPrettyMultilineString)

  //  val a = plane(SymbolicStr("ax"), SymbolicStr("ay"))
  //  val b = plane(SymbolicStr("bx"), SymbolicStr("by"))
  //  val q = a.antiGeometric(b)
  //
  //  println(q.toPrettyMultilineString)
  //  println(q.leftComplement.toPrettyMultilineString)
  //  println(q.rightComplement.toPrettyMultilineString)
}


@main
def rotate3d() = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*

  val a = planeSymbolic("a")
  val b = planeSymbolic("b")
  val q = a.antiGeometric(b)

  println(s"plane a = ${a.toPrettyMultilineString}")
  println(s"plane b = ${b.toPrettyMultilineString}")
  println(s"a antiGeometric b == q == ${a.antiGeometric(b).toPrettyMultilineString}")
  println(s"a antiWedge b == q == ${a.antiWedge(b).toPrettyMultilineString}")

  println(q.toPrettyMultilineString)
  println(q.leftComplement.toPrettyMultilineString)
  println(q.rightComplement.toPrettyMultilineString)

  val qq = rotorSymbolic("q")

  println(s"q^2 = ${qq.antiGeometric(qq).toPrettyMultilineString}")

  val t = translateSymbolic("t")

  println(s"t antiGeometric qq = ${t.antiGeometric(qq).toPrettyMultilineString}")
  println(s"qq antiGeometric t = ${qq.antiGeometric(t).toPrettyMultilineString}")
}

@main
def motor3d(): Unit = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*

  {
    val a = planeSymbolic("a")
    val b = planeSymbolic("b")
    println(
      s"""
         |a = ${a.toPrettyMultilineString}
         |b = ${b.toPrettyMultilineString}
         |a antiGeometric b = ${(a antiGeometric b).toPrettyMultilineString}
         |""".stripMargin)
  }

  val q = rotorSymbolic(0.0, 0.0, 1.0, 0.0)
//  val q = rotorSymbolic("q")
  val t = translateSymbolic(10.0, 0.0, 0.0)
  val p = pointSymbolic("p")
//  val pZero = point(SymbolicStr(0.0), SymbolicStr(0.0), SymbolicStr(0.0))

  //  println(q.antiSandwich(p).toPrettyMultilineString)
  //  println(t.antiSandwich(p).toPrettyMultilineString)

  println(s"t = ${t.toPrettyMultilineString}")
  println(s"q = ${q.toPrettyMultilineString}")
//  println(s"t antiGeometric q = ${t.antiGeometric(q).toPrettyMultilineString}")
  println(s"q antiGeometric t = ${q.antiGeometric(t).toPrettyMultilineString}")

//  println(s"t antiGeometric q sandwich pZero = ${t.antiGeometric(q).antiSandwich(pZero).simplifiedOrdered.toPrettyMultilineString}")
  println(s"t antiGeometric q sandwich p = ${t.antiGeometric(q).antiSandwich(p).simplifiedOrdered.toPrettyMultilineString}")
  //  println(s"q antiGeometric t = ${t.antiGeometric(q).antiSandwich(p).simplifiedOrdered.mapValues(_.flatMapFunctions[SymbolicStr.F] {
  //    case ("*", Seq(SymbolicStr.SymbolStr("p.x"), SymbolicStr.SymbolStr("q.s"), SymbolicStr.SymbolStr("q.s"))) =>
  //      SymbolicStr("p.x") * (SymbolicStr(1.0) - SymbolicStr("q.x") * SymbolicStr("q.x") - SymbolicStr("q.y") * SymbolicStr("q.y") - SymbolicStr("q.z") - SymbolicStr("q.z"))
  //    case (name, elems) => SymbolicStr(name, elems)
  //  }).toPrettyMultilineString}")
}

@main
def motor3dv2(): Unit = Basis.pga3.use {
  import com.github.kright.ga.BasisPGA3.*
//    val q = rotorSymbolic("sin", 0.0, 0.0, "cos")

  val q = MultiVector(
    "xy" -> SymbolicStr("qz") * SymbolicStr("sin"),
    "xz" -> -SymbolicStr("qy") * SymbolicStr("sin"),
    "yz" -> SymbolicStr("qx") * SymbolicStr("sin"),
    "" -> SymbolicStr("cos"),
  ).leftComplement

  val t0 = translateSymbolic("dx", "dy", "dz")
  val t = t0.antiGeometric(t0)
  val p = pointSymbolic("p")
  val pZero = point(SymbolicStr(0.0), SymbolicStr(0.0), SymbolicStr(0.0))

  //  println(q.antiSandwich(p).toPrettyMultilineString)
  //  println(t.antiSandwich(p).toPrettyMultilineString)

  println(s"t = ${t.toPrettyMultilineString}")
  println(s"q = ${q.toPrettyMultilineString}")
  println(s"t antiGeometric q = ${t.antiGeometric(q).toPrettyMultilineString}")
  println(s"q antiGeometric t = ${q.antiGeometric(t).toPrettyMultilineString}")

  //  println(s"t antiGeometric q sandwich pZero = ${t.antiGeometric(q).antiSandwich(pZero).simplifiedOrdered.toPrettyMultilineString}")
  println(s"t antiGeometric q sandwich p = ${t.antiGeometric(q).antiSandwich(p).simplifiedOrdered.toPrettyMultilineString}")
  //  println(s"q antiGeometric t = ${t.antiGeometric(q).antiSandwich(p).simplifiedOrdered.mapValues(_.flatMapFunctions[SymbolicStr.F] {
  //    case ("*", Seq(SymbolicStr.SymbolStr("p.x"), SymbolicStr.SymbolStr("q.s"), SymbolicStr.SymbolStr("q.s"))) =>
  //      SymbolicStr("p.x") * (SymbolicStr(1.0) - SymbolicStr("q.x") * SymbolicStr("q.x") - SymbolicStr("q.y") * SymbolicStr("q.y") - SymbolicStr("q.z") - SymbolicStr("q.z"))
  //    case (name, elems) => SymbolicStr(name, elems)
  //  }).toPrettyMultilineString}")
}

