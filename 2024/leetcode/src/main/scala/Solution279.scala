object Solution279 extends App {
  for(i <- 1 to 10) {
    println(s"$i: ${numSquares(i)}")
  }

  def numSquares(n: Int): Int = {
    val squares = (1 to 100).map(i => i * i).toArray

    val numbers = new Array[Int](10001)

    for (s <- squares) {
      numbers(s) = 1
    }

    for (i <- 1 to n) {
      if (numbers(i) == 0) {
        numbers(i) = squares.takeWhile(_ < i).map(s => numbers(i - s) + 1).min
      }
    }

    numbers(n)
  }
}
