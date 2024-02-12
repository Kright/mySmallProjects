object Solution229 extends App {
  println(majorityElement(Array(2,1,1,3,1,4,5,6)))

  def majorityElement(nums: Array[Int]): List[Int] = {
    var candidate1 = 0
    var candidate2 = 0
    var c1Count = 0
    var c2Count = 0

    for (n <- nums) {
      if (candidate1 == n) {
        c1Count += 1
      } else if (candidate2 == n) {
        c2Count += 1
      } else if (c1Count == 0) {
        candidate1 = n
        c1Count += 1
      } else if (c2Count == 0) {
        candidate2 = n
        c2Count += 1
      }
      else {
        c1Count -= 1
        c2Count -= 1
      }
    }

    List(candidate1, candidate2).distinct.filter(c => nums.count(_ == c) > nums.length / 3)
  }
}
