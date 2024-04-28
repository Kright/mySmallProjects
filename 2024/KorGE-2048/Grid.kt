

class Grid(
    private val width: Int,
    private val height: Int,
    initialValue: Int = 0
) {

    private val data = IntArray(width * height) { initialValue }

    operator fun get(position: Position): Int {
        return data[position.y * width + position.x]
    }

    operator fun set(position: Position, value: Int) {
        data[position.y * width + position.x] = value
    }

    fun copy(): Grid {
        val newGrid = Grid(width, height)
        data.copyInto(newGrid.data)
        return newGrid
    }

    val positions: Sequence<Position>
        get() = sequence {
            for (y in 0 until height) {
                for (x in 0 until width) {
                    yield(Position(x, y))
                }
            }
        }

    val allCells: Sequence<Pair<Position, Int>>
        get() = positions.map { pos -> pos to get(pos) }
}
