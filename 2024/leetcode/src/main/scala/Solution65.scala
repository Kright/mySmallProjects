object Solution65 extends App {

  val numbers = Seq("2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789")
  val nonNumbers = Seq("abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53", "..2")

  numbers.foreach(n => assert(isNumber(n)))
  nonNumbers.foreach(n => assert(!isNumber(n), s"${n} isNotNumber!"))

  def isNumber(s: String): Boolean = {
    val ws = withoutSign(s)
    isUnsignedInteger(ws) || isDecimal(ws)
  }

  private def withoutSign(s: String): String =
    if (s.startsWith("+") || s.startsWith("-")) s.drop(1) else s

  private def isUnsignedInteger(s: String): Boolean = s.nonEmpty && s.forall(_.isDigit)

  private def isSignedInteger(s: String): Boolean = isUnsignedInteger(withoutSign(s))

  private def isDecimal(s: String): Boolean = {
    if (s.isEmpty) return false

    val lastPart = s.dropWhile(_.isDigit).dropWhile(_ == '.').dropWhile(_.isDigit)

    val consumedPart = s.take(s.length - lastPart.length)
    if (consumedPart.isEmpty) return false
    if (!consumedPart.head.isDigit && !consumedPart.last.isDigit) return false
    if (consumedPart.count(_ == '.') > 1) return false

    if (lastPart.isEmpty) return true

    if (lastPart.startsWith("E") || lastPart.startsWith("e")) {
      isSignedInteger(lastPart.drop(1))
    } else false
  }
}
