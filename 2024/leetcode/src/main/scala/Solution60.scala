object Solution60 extends App{
  println(getPermutation(4, 9))

  def getPermutation(n: Int, k: Int): String = {
    def get(s: String, k: Int): String = {
      if (s.length == 1) return s
      val f = factorial(s.length - 1)
      val c = s(k / f)
      s"$c${get(s.filter(_ != c).sorted, k % f)}"
    }

    get((1 to n).mkString(""), k - 1)
  }

  private def factorial(n: Int): Int =
    if (n >= 2) (1 to n).product else 1
}
