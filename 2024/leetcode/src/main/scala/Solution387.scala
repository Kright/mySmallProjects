import scala.collection.mutable

object Solution387 extends App {
  println(firstUniqChar("leetcode"))
  println(firstUniqChar("loveleetcode"))
  println(firstUniqChar("aabb"))

  def firstUniqChar(s: String): Int = {
    val counts = new mutable.HashMap[Char, Int]().withDefault(_ => 0)
    val firstPos = new mutable.HashMap[Char, Int]().withDefault(_ => 0)

    for((c, i) <- s.zipWithIndex) {
      val count = counts(c)
      counts(c) = count + 1
      if (count == 0) {
        firstPos(c) = i
      }
    }

    counts.filter(_._2 == 1).toIndexedSeq.map(c => c._1 -> s.indexOf(c._1)).minByOption(_._2).map(_._2).getOrElse(-1)
  }
}
