import scala.collection.mutable

object Solution76 extends App {
  println(minWindow("ADOBECODEBANC", "ABC"))

  def minWindow(s: String, t: String): String = {
    class CharsCounter {
      private val counts = new mutable.HashMap[Char, Int]()
      private val expectedCounts: Map[Char, Int] = t.groupBy(c => c).map(p => (p._1, p._2.length))
      private val uniqueSymbolsCount = expectedCounts.size
      private var matchingCounts: Int = 0

      for(c <- t) {
        counts(c) = 0
      }

      def isMatching: Boolean = uniqueSymbolsCount == matchingCounts

      def +=(c: Char): Unit = {
        if (!expectedCounts.contains(c)) return
        val expected = expectedCounts(c)
        val real = counts(c)
        if (real + 1 == expected) {
          matchingCounts += 1
        }
        counts(c) = real + 1
      }

      def -=(c: Char): Unit = {
        if (!expectedCounts.contains(c)) return
        val expected = expectedCounts(c)
        val real = counts(c)
        if (real == expected) {
          matchingCounts -= 1
        }
        counts(c) = real - 1
      }
    }

    val counter = new CharsCounter()
    var left = 0
    var right = 0

    var minResult: (Int, Int) = null

    while(right < s.length || counter.isMatching) {
      val isMatching = counter.isMatching

      if (!isMatching && right < s.length) {
        counter += s(right)
        right += 1
      } else {
        if (isMatching && (minResult == null || (minResult._2 - minResult._1) > (right - left))) {
          minResult = (left, right)
        }

        counter -= s(left)
        left += 1
      }
    }

    if (minResult == null) return ""

    s.substring(minResult._1, minResult._2)
  }
}
