
data class Position(
    val x: Int,
    val y: Int
) {
    operator fun plus(other: Position): Position = Position(x + other.x, y + other.y)

    operator fun minus(other: Position): Position = Position(x - other.x, y - other.y)

    fun isWithinBounds(width: Int, height: Int): Boolean =
        x in 0 until width && y in 0 until height
}
