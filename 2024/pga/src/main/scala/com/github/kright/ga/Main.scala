package com.github.kright.ga

@main
def main(): Unit = {

  Basis.pga2.use {
    def point(x: Double, y: Double): MultiVector[Double] =
      MultiVector(Seq(
        BasisBlade.apply("x") -> x,
        BasisBlade.apply("y") -> y,
        BasisBlade.apply("w") -> 1.0,
      ))

    def unitizePoint(v: MultiVector[Double]): MultiVector[Double] =
      v * (1.0 / v.apply(BasisBlade("w") ))

    val p1 = point(1, 0)
    val p2 = point(0, 2)
    val p3 = point(1, 1)

    println(p1 ∧ p2 ∧ p2)
    println(p1 ∧ (p2 ∧ p2))
    println(p2 ∧ p1 ∧ p2)

    printMultiplicationTables()
  }
}

def printMultiplicationTables()(using Basis): Unit = {
  println("geometric")
  println(MultiplicationTable.geometric.toPrettyString(summon[Basis].bladesByOrder))
  println("dot")
  println(MultiplicationTable.dot.toPrettyString(summon[Basis].bladesByOrder))
  println("wedge")
  println(MultiplicationTable.wedge.toPrettyString(summon[Basis].bladesByOrder))
}
