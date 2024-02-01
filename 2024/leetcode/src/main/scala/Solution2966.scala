object Solution2966 {
  def divideArray(nums: Array[Int], k: Int): Array[Array[Int]] = {
    nums.sortInPlace()

    for (i <- nums.indices.by(3)) {
      if (nums(i + 2) - nums(i) > k) {
        return Array()
      }
    }

    val result = new Array[Array[Int]](nums.length / 3)

    for (i <- nums.indices.by(3)) {
      result(i / 3) = nums.view.slice(i, i + 3).toArray
    }

    result
  }
}
