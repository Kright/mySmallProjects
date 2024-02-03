object Solution200 extends App {
  def numIslands(grid: Array[Array[Char]]): Int = {
    val h = grid.length
    val w = grid(0).length
    var islandsCount = 0

    def fillIsland(y: Int, x: Int): Unit = {
      if (x >= 0 && y >= 0 && x < w && y < h && grid(y)(x) == '1') {
        grid(y)(x) = '0'
        fillIsland(y, x - 1)
        fillIsland(y, x + 1)
        fillIsland(y - 1, x)
        fillIsland(y + 1, x)
      }
    }

    for (y <- 0 until h) {
      for (x <- 0 until w) {
        if (grid(y)(x) == '1') {
          islandsCount += 1
          fillIsland(y, x)
        }
      }
    }

    islandsCount
  }
}

