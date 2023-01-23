import org.apache.pdfbox.pdmodel.{PDDocument, PDPageTree}
import org.apache.pdfbox.text.{PDFTextStripper, PDFTextStripperByArea, TextPosition}

import java.awt.Rectangle
import java.awt.geom.Rectangle2D
import collection.convert.ImplicitConversionsToScala.*
import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.util
import scala.collection.mutable.ArrayBuffer
import scala.util.{Failure, Success, Try}

@main
def main(): Unit =
  val dir = new File("") // add here path to directory with files
  require(dir.exists())

  dir.listFiles().filter(_.getName.toLowerCase.endsWith(".pdf")).sorted.foreach{ pdfFile =>
    val csvFile = new File(pdfFile.getParent, pdfFile.getName + ".csv")
    println(s"convert $pdfFile => $csvFile")
    processFile(pdfFile, csvFile)
  }

def processFile(pdfFile: File, csvFile: File): Unit =
  val rows = textRows(pdfFile).map(_.withLimitHeight(28))

  val extractors = Seq(
    ColumnExtractor("datumPrijema", 26, isLeftCorner = true, s => {require(s.size == 1); convertDate(s.head.text)}),
    ColumnExtractor("datumIzvrsenja", 90, isLeftCorner = true, s => {require(s.size == 1); convertDate(s.head.text)}),
    ColumnExtractor("brojCartice", 152, isLeftCorner = true, s => {s.headOption.map(_.text).getOrElse("")}),
    ColumnExtractor("opis", 206, isLeftCorner = true, s => {s.map(_.text).mkString(" ")}),
    ColumnExtractor("stanje", 546 + 36, isLeftCorner = false, s => {require(s.size == 1); s.head.text}),
    ColumnExtractor("uplata", 516 + 14, isLeftCorner = false, s => {require(s.size == 1); s.head.text}),
    ColumnExtractor("isplata", 441 + 36, isLeftCorner = false, s => {require(s.size == 1); s.head.text}),
    ColumnExtractor("iznosOrigValuta", 371 + 53, isLeftCorner = false, s => s.headOption.map(_.text).getOrElse("")),
    ColumnExtractor("iznosRefValuta", 345 + 16, isLeftCorner = false, s => s.head.text)
  )

  val csvText = new StringBuilder()
  csvText.append(extractors.map(_.name).mkString(", ")).append("\n")
  csvText.append(rows.map { row =>
    extractors.map(_.extract(row)).map(_.filter(c => c != '\n' && c != ',')).mkString(", ")
  }.mkString("\n"))

  Files.write(csvFile.toPath, csvText.toString().getBytes(StandardCharsets.UTF_8))

class ColumnExtractor(val name: String,
                      val x: Int,
                      val isLeftCorner: Boolean,
                      val prepare: Seq[TextWithPos] => String):
  def extract(row: Row): String =
    try {
      prepare(if (isLeftCorner) row.arr.filter(_.isLeftCorner(x, eps = 10)) else row.arr.filter(_.isRightCorner(x, eps = 10)))
    } catch {
      case ex: Exception => throw new RuntimeException(s"can't extract $name from row = $row", ex)
    }


def textRows(file: File): Seq[Row] = {
  val result = new ArrayBuffer[ArrayBuffer[TextWithPos]]()

  class MyWorkaroundStripper extends PDFTextStripper {
    override def writeString(text: String, textPositions: util.List[TextPosition]): Unit = {
      if (text.length == 20 && isDate(text.substring(0, 10)) && isDate(text.substring(10, 20))) {
        result += new ArrayBuffer()
      }
      if (result.nonEmpty) {
        result.last ++= separate(text, textPositions, maxWidth = 10)
      }
    }
  }

  val stripper = new MyWorkaroundStripper()

  stripper.setShouldSeparateByBeads(true)
  stripper.setIndentThreshold(0.0)
  val doc = PDDocument.load(file)
  stripper.getText(doc)

  result.map(arr => new Row(arr.toSeq)).toSeq
}

case class TextWithPos(text: String, pos: Rectangle):
  def combine(tp: TextWithPos): TextWithPos = TextWithPos(text + tp.text, combine(pos, tp.pos))

  def combine(r1: Rectangle, r2: Rectangle): Rectangle =
    val rectangles = Seq(r1, r2)
    val minX = rectangles.map(r => r.x).min
    val minY = rectangles.map(r => r.y).min
    val maxX = rectangles.map(r => r.x + r.width).max
    val maxY = rectangles.map(r => r.y + r.height).max
    new Rectangle(minX, minY, maxX - minX, maxY - minY)

  def isLeftCorner(x: Int, eps: Int): Boolean = math.abs(pos.x - x) <= eps

  def isRightCorner(x: Int, eps: Int): Boolean = math.abs(pos.x + pos.width - x) <= eps

class Row(val arr: Seq[TextWithPos]):
  require(arr.nonEmpty)

  def withLimitHeight(maxHeight: Int): Row =
    val firstY = arr.head.pos.y
    Row(arr.filter(t => Math.abs(firstY - t.pos.y) <= maxHeight))

  override def toString: String = arr.mkString("[", ", ", "]")


def separate(text: String, textPositions: Iterable[TextPosition], maxWidth: Int): Seq[TextWithPos] =
  require(text.length == textPositions.size)
  val results = new ArrayBuffer[TextWithPos]()

  for ((char, pos) <- text.zip(textPositions)) {
    val rect = new Rectangle(Math.round(pos.getX), Math.round(pos.getY), Math.round(pos.getWidth), Math.round(pos.getHeight))
    val current = TextWithPos(s"$char", rect)

    if (results.nonEmpty && Math.abs(results.last.pos.getMaxX - current.pos.getMinX) < maxWidth) {
      val last = results.remove(results.size - 1)
      results += last.combine(current)
    } else {
      results += current
    }
  }

  results.toSeq

def isDate(s: String): Boolean =
  if (s.length != 10) return false
  if (s(2) != '.' || s(5) != '.') return false
  Seq(0, 1, 3, 4, 6, 7, 8, 9).forall(i => s(i).isDigit)

def convertDate(s: String): String =
  require(isDate(s))
  s"${s.substring(6, 10)}.${s.substring(3, 5)}.${s.substring(0, 2)}"
