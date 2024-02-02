import scala.collection.IterableOnce.iterableOnceExtensionMethods
import scala.collection.mutable.ArrayBuffer

object Solution51 extends App {

  val s = System.nanoTime()
  println(solveNQueens(8))
  println(System.nanoTime() - s)

  def solveNQueens(n: Int): List[List[String]] = {
    class Board {
      private var currentLevel: Int = 0
      private val positions = new Array[Int](n + 1)
      private val lines = (0 until n).map(p => (0 until n).map(i => if (i == p) "Q" else ".").mkString(""))

      def canStepForward(): Boolean = {
        val level = currentLevel
        val pos = positions(level)
        var j = 0
        while (j < level) {
          val dPos = positions(j) - pos
          if (dPos == (j - level) || dPos == (level - j) || dPos == 0) return false
          j += 1
        }
        true
      }

      def stepBack(): Unit = {
        val l = currentLevel
        positions(currentLevel) = 0
        currentLevel -= 1
        if (currentLevel >= 0) {
          positions(currentLevel) += 1
        }
      }

      def makeRepr(): List[String] =
        (0 until n).map(i => lines(positions(i))).toList

      def searchAll(): List[List[String]] = {
        val result = new ArrayBuffer[List[String]]()

        while (currentLevel >= 0) {
          if (currentLevel == n) {
            result += makeRepr()
            stepBack()
          } else if (positions(currentLevel) == n) {
            stepBack()
          } else if (canStepForward()) {
            currentLevel += 1
          } else {
            positions(currentLevel) += 1
          }
        }

        result.toList
      }
    }

    new Board().searchAll()
  }
}
