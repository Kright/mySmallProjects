import scala.collection.mutable.ArrayBuffer

object Solution1 {
  def twoSum(nums: Array[Int], target: Int): Array[Int] = {
    val sortedNums = nums.sorted

    val (n, m) = twoSumForSorted(sortedNums, target)

    val result = ArrayBuffer[Int]()

    for (i <- nums.indices) {
      val v = nums(i)
      if (v == n || v == m) {
        result += i
      }
    }

    result.toArray
  }

  private def twoSumForSorted(sortedNums: Array[Int], target: Int): (Int, Int) = {
    var low = 0
    var up = sortedNums.length - 1

    while(true) {
      val sum = sortedNums(low) + sortedNums(up)
      if (sum == target) {
        return (sortedNums(low), sortedNums(up))
      }
      if (sum > target) {
        up -= 1
      } else {
        low += 1
      }
    }

    ???
  }
}