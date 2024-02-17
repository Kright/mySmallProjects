package com.github.kright.ga

@main
def main(): Unit = {
  println("hello world")

  Basis.ga2.use{
    printMultiplicationTables()
  }

  Basis.ga3.use {
    printMultiplicationTables()
  }

  Basis.pga2.use {
    val x = BasisBlade("x")
    val y = BasisBlade("y")

//    println(x)
//    println(y)

    printMultiplicationTables()
  }

  Basis.pga3.use {
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
