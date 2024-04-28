import korlibs.event.*
import korlibs.korge.*
import korlibs.korge.view.*
import korlibs.image.color.*
import korlibs.image.text.*
import korlibs.korge.input.*
import korlibs.korge.view.align.*
import korlibs.math.geom.*

suspend fun main() = Korge(windowSize = Size(512, 512), bgcolor = Colors["#faf8ef"]) {
    val gridSize = 4
    val tileSize = 100
    val padding = 10

    val game = Game(gridSize)
    val tilesView = container {}

    // Отрисовка игрового поля
    fun renderGrid() {
        tilesView.removeChildren()
        for((pos, value) in game.grid.allCells) {
            val tile = roundRect(Size(tileSize, tileSize), RectCorners.ONE * 10.0, fill = getTileColor(value))
                .xy(pos.x * (tileSize + padding) + padding, pos.y * (tileSize + padding) + padding)

            tile.addChild(TextBlock(RichTextData(value.toString(), textSize = 32.0, color = Colors.BLACK), align = TextAlignment.MIDDLE_CENTER) )

            tilesView.addChild(tile)
        }
    }

    // Обновление отображения после хода
    fun updateGame(direction: Direction) {
        println("updateGame(${direction})")
        if (game.makeMove(direction)) {
            game.addRandomTile()
        }
        renderGrid()
        if (game.isGameOver()) {
            text("Game Over", textSize = 48.0, color = Colors.RED)
                .centerOnStage()
        }
    }

    // Отображение начального состояния
    addChild(tilesView)
    game.addRandomTile()
    game.addRandomTile()
    renderGrid()

    // Управление
    keys {
        justDown(Key.DOWN, Key.S) { updateGame(Direction.DOWN) }
        justDown(Key.UP, Key.W) { updateGame(Direction.UP) }
        justDown(Key.RIGHT, Key.D) { updateGame(Direction.RIGHT) }
        justDown(Key.LEFT, Key.A) { updateGame(Direction.LEFT) }
    }
}

// Цвет плиток
fun getTileColor(value: Int) = when (value) {
    2 -> Colors["#eee4da"]
    4 -> Colors["#ede0c8"]
    8 -> Colors["#f2b179"]
    16 -> Colors["#f59563"]
    32 -> Colors["#f67c5f"]
    64 -> Colors["#f65e3b"]
    128 -> Colors["#edcf72"]
    256 -> Colors["#edcc61"]
    512 -> Colors["#edc850"]
    1024 -> Colors["#edc53f"]
    2048 -> Colors["#edc22e"]
    else -> Colors["#cdc1b4"]
}

