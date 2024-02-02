object Solution1291{
  def sequentialDigits(low: Int, high: Int): List[Int] = {
    val all = for (
      len <- 1 to 9;
      start <- 1 to 10 - len
    ) yield (start until (start + len)).mkString("")

    all.map(_.toInt).dropWhile(_ < low).takeWhile(_ <= high).toList
  }
}
