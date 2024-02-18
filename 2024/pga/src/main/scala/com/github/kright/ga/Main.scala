package com.github.kright.ga

@main
def main(): Unit = {
  println("hello world")

  Basis.ga3.use{
    println("geometric")
    println(basis.geometric.toPrettyString(basis.bladesByOrder))
    println("wedge")
    println(basis.wedge.toPrettyString(basis.bladesByOrder))
    println("dot")
    println(basis.dot.toPrettyString(basis.bladesByOrder))
  }

  Basis.ga3.use {
    val a = MultiVector(basis.blades.zip((1 to basis.bladesCount).map(_.toDouble)).filter(_._1.order == 1))
    val b = MultiVector(basis.blades.zip((basis.bladesCount + 1 to basis.bladesCount * 2).map(_.toDouble)).filter(_._1.order == 1))
    val c = MultiVector(basis.blades.zip((basis.bladesCount * 2 + 1 to basis.bladesCount * 3).map(_.toDouble)).filter(_._1.order == 1))

    println(s"a = ${a}, mag = ${a.magnitude}")
    println(s"b = ${b}, mag = ${b.magnitude}")

    println(basis.geometric.toPrettyString(basis.bladesByOrder))

    println("geometric")
    println(a ⟑ b)
    println(b ⟑ a)

    println("wedge")
    println(a ∧ b)
    println(b ∧ a)

    println("dot")
    println(a ⋅ (b ⟑ c))
    println((b ⟑ c) ⋅ a)
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
