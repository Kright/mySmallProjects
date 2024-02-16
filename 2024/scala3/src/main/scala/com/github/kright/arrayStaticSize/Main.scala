package com.github.kright.arrayStaticSize

import scala.compiletime.constValue

@main
def main(): Unit = {
  Main().main()
}

class Main {
  def main(): Unit = {
    val a = ArrayAnySize[Int, 3]()
    val b = ArrayAnySize[Int, 5]()
    val c = a ++ b
    val d = c.strictZip(a ++ b)
    //    val e = c.strictZip(a) // will be error
  }
}