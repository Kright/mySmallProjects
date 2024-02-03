object Solution41 extends App {
  println(firstMissingPositive(Array(1, 1)))

  def firstMissingPositive(nums: Array[Int]): Int = {
    for (i <- nums.indices) {
      setInPlace(nums, i)
    }

    for (i <- nums.indices) {
      if (nums(i) != i + 1)
        return i + 1
    }

    nums.length + 1
  }

  private def setInPlace(nums: Array[Int], pos: Int): Unit = {
    while (true) {
      val n = nums(pos)
      if (n <= 0 || n > nums.length) return

      val newPos = n - 1
      if (newPos == pos) return

      if (nums(newPos) == n) return

      swap(nums, newPos, pos)
    }
  }

  private def swap(arr: Array[Int], i: Int, j: Int): Unit = {
    val t = arr(i)
    arr(i) = arr(j)
    arr(j) = t
  }
}
