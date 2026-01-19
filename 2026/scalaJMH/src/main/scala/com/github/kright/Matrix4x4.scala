package com.github.kright

class Matrix4x4 {
  val data: Array[Double] = Array.ofDim[Double](16)

  def apply(row: Int, column: Int): Double = data(row * 4 + column)

  def update(row: Int, column: Int, value: Double): Unit = {
    data(row * 4 + column) = value
  }

  def :=(other: Matrix4x4): Unit = {
    Array.copy(other.data, 0, data, 0, 16)
  }

  def copy(): Matrix4x4 = {
    val m = Matrix4x4()
    m := this
    m
  }

  override def toString: String = {
    data.mkString("[", ", ", "]")
  }
}


object Matrix4x4 {
  def id(): Matrix4x4 = {
    val m = Matrix4x4()
    for (i <- 0 to 3) {
      m(i, i) = 1.0
    }
    m
  }

  def random(): Matrix4x4 = {
    val m = Matrix4x4()
    for (i <- 0 until 16) {
      m.data(i) = scala.util.Random.nextDouble()
    }
    m
  }
}


def multiply(a: Matrix4x4, b: Matrix4x4, result: Matrix4x4): Unit = {
  for (row <- 0 to 3) {
    for (column <- 0 to 3) {
      var sum = 0.0
      for (i <- 0 to 3) {
        sum += a(row, i) * b(i, column)
      }
      result(row, column) = sum
    }
  }
}


def multiplyFastLoop(a: Matrix4x4, b: Matrix4x4, result: Matrix4x4): Unit = {
  fastLoop(4) { row =>
    fastLoop(4) { column =>
      var sum = 0.0
      fastLoop(4) { i =>
        sum += a(row, i) * b(i, column)
      }
      result(row, column) = sum
    }
  }
}


inline def fastLoop(count: Int)(body: Int => Unit): Unit = {
  var i = 0
  while (i < count) {
    body(i)
    i += 1
  }
}
