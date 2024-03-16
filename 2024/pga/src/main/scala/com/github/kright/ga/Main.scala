package com.github.kright.ga

type M = MultiVector[Double]

@main
def main(): Unit = {
//  Basis.pga2.use {
//    val center = MultiVector[Symbolic]("w" -> Symbolic.Constant(1.0))
//    val dir = MultiVector[Symbolic](basis.bladesByOrder.filter(b => Set(0, 2, 3).contains(b.order)).map(b => b -> (if (b.bits == 0) Symbolic.Symbol("s") else Symbolic.Symbol(s"$b"))))
//
//    println(s"dir = ${dir}")
//    println((dir ⟇ center ⟇ dir).mapValues(_.simplified).toMultilineString)
//  }
  Basis.pga3.use{
    printMultiplicationTables()
  }

  return

  Basis.pga2.use {
    def point(x: Double, y: Double): M =
      MultiVector(
        "x" -> x,
        "y" -> y,
        "w" -> 1.0,
      )

    def motor(first: M, second: M): M =
      first ⟑ second

    def unitizePoint(v: M): M =
      v * (1.0 / v(BasisBlade("w")))

    def setLen1(v: M): M =
      v / math.sqrt((v.dot(v))(basis.scalarBlade))

    val p1 = point(1, 0)
    val p2 = point(0, 2)
    val p3 = point(1, 1)

    val x1 = point(1, 0)
    val y1 = point(0, 1)
    val center = point(0, 0)

    for (blade <- basis.blades) {
      val d = MultiVector(Seq(BasisBlade("x") -> 1.0, blade -> 1.0))
      val p = MultiVector("w" -> 1.0)

      println(
        s"""d = ${d}
           |d ⟑ p ⟑ d = ${(d ⟑ p ⟑ d).filter((b, v) => v != 0.0)}
           |""".stripMargin)

      println(
        s"""d = ${d}
           |d ⟇ p ⟇ d = ${(d ⟇ p ⟇ d).filter((b, v) => v != 0.0)}
           |""".stripMargin)
    }

    //
    //    val dir1 = MultiVector("x" -> 1.0, "y" -> 0.0, "xy" -> 1.0)
    //    val dir2 = MultiVector("x" -> 1.0, "y" -> 0.0, "w" -> 1.0)
    ////    val dir2 = setLen1(MultiVector("x" -> 1.0, "y" -> 1.0))
    //
    //    for (p <- Seq(center); d <- Seq(dir1, dir2)) {
    //      println(
    //        s"""p = ${p}
    //           |d = ${d}
    //           |-d p d = ${-d ⟑ p ⟑ d}
    //           |""".stripMargin)
    //    }

    /*
    println(
      s"""dir1 = ${dir1}
         |dir2 = ${dir2}
         |dir1 ⟑ dir2 = ${dir1 ⟑ dir2}
         |dir2 ⟑ dir1 = ${dir2 ⟑ dir1}
         |""".stripMargin)

    println(
      s"""rotated(center) = ${dir1 ⟑ dir2 ⟑ center ⟑ dir2 ⟑ dir1}
         |rotated(x1) = ${dir1 ⟑ dir2 ⟑ x1 ⟑ dir2 ⟑ dir1}
         |rotated(y1) = ${dir1 ⟑ dir2 ⟑ y1 ⟑ dir2 ⟑ dir1}
         |""".stripMargin)
     */
  }

  //    printMultiplicationTables()
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
