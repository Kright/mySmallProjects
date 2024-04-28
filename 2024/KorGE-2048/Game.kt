import kotlin.random.Random

class Game(val size: Int) {
    val grid = Grid(size, size, 0)

    fun addRandomTile() {
        val emptyCells = grid.allCells.filter { it.second == 0 }.toList()
        if (emptyCells.isNotEmpty()) {
            val (pos, _) = emptyCells.random()
            grid[pos] = if (Random.nextDouble() < 0.9) 2 else 4
        }
    }

    fun makeMove(direction: Direction): Boolean {
        return moveGrid(grid, mapPositions(direction))
    }

    fun canMakeMove(direction: Direction): Boolean {
        return moveGrid(grid.copy(), mapPositions(direction))
    }

    private fun mapPositions(direction: Direction): (Position) -> Position =
        when (direction) {
            Direction.UP -> { p -> p}
            Direction.DOWN -> { p -> Position(p.x, size - 1 - p.y) }
            Direction.LEFT -> { p -> Position(p.y, p.x) }
            Direction.RIGHT -> { p -> Position(size - 1 - p.y, p.x) }
        }

    private fun moveGrid(grid: Grid, mapPositions: (Position) -> Position): Boolean {
        var moved = false

        for(x in 0 until size) {
            val rowPositions = (0 until size)
                .map { y -> mapPositions(Position(x, y)) }

            val row = rowPositions.map { grid[it] }
            val mergedTiles = mergeTiles(row.filter { it != 0 })
            require(mergedTiles.size == size)

            rowPositions.zip(mergedTiles).forEach { (pos, v) ->
                grid[pos] = v
            }

            moved = moved or (mergedTiles != row)
        }
        return moved
    }

    private fun mergeTiles(row: List<Int>): List<Int> {
        val merged = mutableListOf<Int>()
        var skip = false
        for (i in row.indices) {
            if (skip) {
                skip = false
                continue
            }
            if (i < row.size - 1 && row[i] == row[i + 1]) {
                merged.add(row[i] * 2)
                skip = true
            } else {
                merged.add(row[i])
            }
        }
        while (merged.size < size) merged.add(0)
        return merged
    }

    fun isGameOver(): Boolean {
        return grid.allCells.all { it.second != 0 } &&
            !listOf(Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT).any { canMakeMove(it) }
    }
}
