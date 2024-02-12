object Solution647 extends App {
  val t = System.nanoTime()
  println(countSubstrings("a".repeat(1000)))
  println(System.nanoTime() - t)

  def countSubstrings(s: String): Int = {
    def isPalindrome(start: Int, len: Int) = (0 until (len / 2)).forall(i => s(start + i) == s(start + len - 1 - i))

    (0 until s.length).map{ start =>
      (1 to (s.length - start)).count{length =>
        isPalindrome(start, length)
        }
    }.sum
  }
}
