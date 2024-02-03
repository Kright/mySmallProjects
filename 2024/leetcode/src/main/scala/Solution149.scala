import scala.collection.mutable

/**
 * geometric algebra rocks, see https://rigidgeometricalgebra.org/wiki/index.php?title=Join_and_meet
 */
object Solution149 extends App {
  println(maxPoints(Array(Array(1, 1), Array(3, 2), Array(5, 3), Array(4, 1), Array(2, 3), Array(1, 4))))

  case class Line(xy: Int, xw: Int, yw: Int)

  object Line {
    def wedge(p1x: Int, p1y: Int, p2x: Int, p2y: Int): Line = {
      val xy = p1x * p2y - p1y * p2x
      val xw = p1x - p2x
      val yw = p1y - p2y

      val s = sign(xy, xw, yw)
      val d = s * gcd(gcd(math.abs(xy), math.abs(xw)), math.abs(yw))
      Line(xy / d, xw / d, yw / d)
    }

    private def sign(xy: Int, xw: Int, yw: Int): Int = {
      if (xy != 0) return if (xy > 0) 1 else -1
      if (xw != 0) return if (xw > 0) 1 else -1
      if (yw != 0) return if (yw > 0) 1 else -1
      0
    }

    private def gcd(a: Int, b: Int): Int = {
      require(a >= 0)
      require(b >= 0)

      if (a < b) return gcd(b, a)

      if (b == 0) return a
      gcd(b, a % b)
    }
  }

  def maxPoints(points: Array[Array[Int]]): Int = {
    if (points.length < 2) return points.length

    var result: Int = 0
    for ((p1, i) <- points.zipWithIndex) {
      val lines = new mutable.HashMap[Line, Int]().withDefault(_ => 0)

      for (j <- points.indices if i != j) {
        val p2 = points(j)
        val line = Line.wedge(p1(0), p1(1), p2(0), p2(1))
        lines(line) += 1
      }

      val (_, otherPointsOnBestLine) = lines.maxBy { case (_, count) => count }
      result = math.max(result, otherPointsOnBestLine + 1)
    }

    result
  }
}
