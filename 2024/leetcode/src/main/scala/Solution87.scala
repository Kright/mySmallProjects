import scala.collection.mutable

object Solution87 extends App {

  val t = System.nanoTime()
  println(isScramble("great", "rgeat"))
  println(isScramble("abb", "bab"))
  println(isScramble("eebaacbcbcadaaedceaaacadccd", "eadcaacabaddaceacbceaabeccd"))
  println(isScramble("oatzzffqpnwcxhejzjsnpmkmzngneo", "acegneonzmkmpnsjzjhxwnpqffzzto"))
  val t2 = System.nanoTime()
  println(t2 - t)

  def isScramble(s1: String, s2: String): Boolean = {
    val memo = new mutable.HashMap[String, Boolean]()

    def memoizedIsScramble(s1: String, s2: String): Boolean = {
      require(s1.length == s2.length)
      val args = s1 + s2

      if (memo.contains(args)) {
        return memo(args)
      }

      if (s1.isEmpty) return true
      if (s1 == s2) return true
      if (s1.length == 1) return false

      if (s1.sorted != s2.sorted) {
        memo(args) = false
        return false
      }

      for (i <- 1 until s1.length) {
        val (x, y) = s1.splitAt(i)

        if (memoizedIsScramble(x, s2.take(x.length)) && memoizedIsScramble(y, s2.drop(x.length))) {
          memo(args) = true
          return true
        }
        if (memoizedIsScramble(y, s2.take(y.length)) && memoizedIsScramble(x, s2.drop(y.length))) {
          memo(args) = true
          return true
        }
      }

      memo(args) = false
      false
    }

    memoizedIsScramble(s1, s2)
  }


}
