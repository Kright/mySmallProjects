

extension (s: String)
  def -(suffix: String): String =
    require(s.endsWith(suffix))
    s.substring(0, s.length - suffix.length)