object Solution2149 extends App {
  println(rearrangeArray(Array(3, 1, -2, -5, 2, -4)).mkString(", "))

  def rearrangeArray(nums: Array[Int]): Array[Int] = {
    val (pos, neg) = nums.partition(_ > 0)
    pos.zip(neg).flatMap{ case (a, b) => List(a, b) }
  }
}
