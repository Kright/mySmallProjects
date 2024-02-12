import scala.collection.mutable.ArrayBuffer

object Solution368 extends App {
  def largestDivisibleSubset(nums: Array[Int]): List[Int] = {
    nums.sortInPlace()

    val results = new ArrayBuffer[List[Int]]()
    results += List.empty

    for((n, i) <- nums.zipWithIndex) {
      val largestList = results.filter(lst => lst.forall(n % _ == 0)).maxBy(_.length)
      results += n :: largestList
    }

    results.maxBy(_.length)
  }
}
