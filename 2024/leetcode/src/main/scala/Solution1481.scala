object Solution1481 {

  def findLeastNumOfUniqueInts(arr: Array[Int], k: Int): Int = {
    arr.groupBy(i => i).toSeq.sortBy(_._2.size).flatMap(_._2).drop(k).distinct.size
  }
}
