object Solution169 extends App {
  /**
   * https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_majority_vote_algorithm
   * https://www.cs.utexas.edu/~moore/best-ideas/mjrty/
   */
  def majorityElement(nums: Array[Int]): Int = {
    var element: Int = 0
    var count = 0
    for (n <- nums) {
      if (count == 0) {
        element = n
        count = 1
      } else {
        if (element == n) count += 1
        else count -= 1
      }
    }
    element
  }
}
