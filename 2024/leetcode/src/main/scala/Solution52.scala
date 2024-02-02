object Solution52 extends App {

  val s = System.nanoTime()
  println(totalNQueens(8))
  println(System.nanoTime() - s)

  def totalNQueens(n: Int): Int = {
    class Board {
      private var currentLevel: Int = 0
      private val positions = new Array[Int](n + 1)

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

      def searchAll(): Int = {
        var result = 0

        while (currentLevel >= 0) {
          if (currentLevel == n) {
            result += 1
            stepBack()
          } else if (positions(currentLevel) == n) {
            stepBack()
          } else if (canStepForward()) {
            currentLevel += 1
          } else {
            positions(currentLevel) += 1
          }
        }

        result
      }
    }

    new Board().searchAll()
  }
}