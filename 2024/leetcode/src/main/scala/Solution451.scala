object Solution451 extends App {
  def frequencySort(s: String): String = {
    val counts = s.groupMapReduce(c => c)(_ => 1)(_ + _)
    counts.toList.sortBy { case (c, count) => -count }.map { case (c, count) => repeat(c, count)}.mkString("")
  }

  private def repeat(c: Char, count: Int): String =
    (1 to count).map(_ => c).mkString("")
}