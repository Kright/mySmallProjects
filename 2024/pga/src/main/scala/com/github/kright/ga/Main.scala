package com.github.kright.ga

type M = MultiVector[Double]

@main
def main(): Unit = {
  Basis.pga2.use {
    val center = MultiVector[Symbolic]("xy" -> Symbolic.Constant(1.0))
    val dir = MultiVector[Symbolic](basis.bladesByOrder.map(b => b -> (if (b.bits == 0) Symbolic.Symbol("s") else Symbolic.Symbol(s"$b"))))

    println(s"dir = ${dir}")
    println(dir.wedge(center).wedge(dir))
    println(dir.wedge(center).wedge(dir).mapValues(_.simplified))

    printMultiplicationTables()
  }

  Basis.ga3.use {
    val center = MultiVector[Symbolic]("xy" -> Symbolic.Constant(1.0))
    val dir = MultiVector[Symbolic](basis.bladesByOrder.map(b => b -> (if (b.bits == 0) Symbolic.Constant(1.0) else Symbolic.Symbol(s"$b"))))

    println(dir.wedge(center).wedge(dir))
    println(dir.wedge(center).wedge(dir).map((b, v) => v.simplified))

    return
  }


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

    println(p1 ∧ p2 ∧ p2)
    println(p1 ∧ (p2 ∧ p2))
    println(p2 ∧ p1 ∧ p2)

    for (blade <- basis.blades) {
      val d = MultiVector(Seq(BasisBlade("x") -> 1.0, blade -> 1.0))
      val p = MultiVector("w" -> 1.0)

      println(
        s"""d = ${d}
           |d p d = ${(d ⟑ p ⟑ d).filter((b, v) => v != 0.0)}
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
  println("geometric")
  println(MultiplicationTable.fromRule(_.geometric).toPrettyString(summon[Basis].bladesByOrder))
  println("geometricAntiproduct")
  println(MultiplicationTable.fromRule(_.geometricAntiproduct).toPrettyString(summon[Basis].bladesByOrder))
  println("dot")
  println(MultiplicationTable.fromRule(_.dot).toPrettyString(summon[Basis].bladesByOrder))
  println("wedge")
  println(MultiplicationTable.fromRule(_.wedge).toPrettyString(summon[Basis].bladesByOrder))
}
