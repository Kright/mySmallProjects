import java.io
import java.io.{Closeable, File}
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import scala.jdk.CollectionConverters.*
import scala.util.{Failure, Success, Try}

extension (file: File)
  def readLines(): Seq[String] =
    Files.readAllLines(file.toPath).asScala.toSeq

  def writeLines(text: String): Unit =
    Files.write(file.toPath, text.getBytes(StandardCharsets.UTF_8))

  def writeLines(body: StringBuilder => Unit): Unit =
    val sb = StringBuilder()
    body(sb)
    writeLines(sb.toString())

  def readCsv(): (Seq[String], Seq[Map[String, String]]) =
    val lines = readLines()

    def splitLine(s: String) = s.split(",").map(_.trim)

    val names = splitLine(lines.head)
    (names, lines.tail.filterNot(_.isBlank).map { line =>
      val words = splitLine(line)
      require(words.size == names.size)
      names.zip(words).toMap
    })

  def writeCsv(names: Seq[String], lines: Iterable[Map[String, String]]): Unit =
    writeLines { sb =>
      sb.append(names.mkString(", "))
      sb.append("\n")
      for (line <- lines) {
        sb.append(names.map(name => line(name)).mkString(","))
        sb.append("\n")
      }
    }

extension [T <: Closeable](c: T)
  def use[R](f: (T) => R): R =
    Try {
      f(c)
    } match
      case Success(result) =>
        c.close()
        result
      case Failure(exception) =>
        Try {
          c.close()
        }.failed.foreach { closeErr =>
          exception.addSuppressed(closeErr)
        }
        throw exception
