package com.github.kright.ga

@main
def main(): Unit = {
  println("hello world")

  Basis.ga2.use {
    val a = MultiVector(basis.blades.zip((1 to basis.bladesCount).map(_.toDouble)).filter(_._1.order == 1))
    val b = MultiVector(basis.blades.zip((basis.bladesCount + 1 to basis.bladesCount * 2).map(_.toDouble)).filter(_._1.order == 1))

    println(s"a = ${a}, mag = ${a.magnitude}")
    println(s"b = ${b}, mag = ${b.magnitude}")
    println(s"a geometric b = ${a.geometric(b)}, mag = ${a.geometric(b).magnitude} xz ${a.magnitude * b.magnitude} ")
    println(s"a dot b = ${a.dot(b)}")
    println(s"a wedge b = ${a.wedge(b)}")

    println(s"a ⟑ a = ${a ⟑ a}")
    println(s"a ⟑ a* = ${a ⟑ a.map((b, v) => if (b.order % 2 == 0) v else -v)}")
    println(s"a ⟑ a* = ${a ⟑ a.map((b, v) => if (b.order == 0) v else -v)}")
    println()
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
