object Solution1043 extends App{

  println(maxSumAfterPartitioning(Array(1, 15, 7, 9, 2, 5, 10), 3))

  def maxSumAfterPartitioning(arr: Array[Int], k: Int): Int = {
    val bestValues = new Array[Int](arr.size)

    def getBest(i: Int): Int = if (i >= 0) bestValues(i) else 0

    for(i <- arr.indices) {
      var curMax = arr(i)
      var curBest = curMax + getBest(i - 1)

      for(j <- (i - 1) to math.max(-1, i - k) by(-1)) {
        curBest = math.max(curBest, (i - j) * curMax + getBest(j))
        if (j >= 0) {
          curMax = math.max(curMax, arr(j))
        }
      }

      bestValues(i) = curBest
    }

    bestValues.last
  }
}
