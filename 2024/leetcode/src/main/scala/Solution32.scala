object Solution32 extends App {
  println(longestValidParentheses("(()"))
  println(longestValidParentheses(")()())"))
  println(longestValidParentheses(""))
  println(longestValidParentheses("()"))

  def longestValidParentheses(s: String): Int = {
    math.max(findLongestValid(s), findLongestValid(s.reverse.map(c => if (c == ')') '(' else ')')))
  }

  private def findLongestValid(s: String): Int = {
    var level: Int = 0
    var startPos: Int = 0
    var longestValid: Int = 0

    for ((c, i) <- s.zipWithIndex) {
      if (c == '(') {
        level += 1
      } else { // c == ')'
        if (level > 0) {
          level -= 1
          if (level == 0) {
            longestValid = math.max(longestValid, i + 1 - startPos)
          }
        } else if (level == 0) {
          longestValid = math.max(longestValid, i - startPos)
          startPos = i + 1
        }
      }
    }

    longestValid
  }
}
