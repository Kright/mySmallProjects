package com.github.kright

import java.io.File
import java.nio.file.{Files, StandardOpenOption}
import java.time.format.DateTimeFormatter
import java.time.temporal.TemporalAdjusters
import java.time.{DayOfWeek, LocalDate, LocalDateTime, ZoneId}
import java.util.Locale
import scala.collection.mutable
import scala.language.implicitConversions
import scala.util.Random
import scala.util.chaining.*

object Utils {
  def getRandom[T](s: Seq[T]): Option[T] =
    if (s.isEmpty) None
    else Some(s(Random.nextInt(s.size)))

  def count[T](s: IterableOnce[T]): mutable.HashMap[T, Int] =
    new mutable.HashMap[T, Int]().tap { map =>
      s.iterator.foreach { t =>
        map(t) = 1 + map.getOrElse(t, 0)
      }
    }

  def unixTime(): Long = System.currentTimeMillis() / 1000L

  def getPrevMonday: Long = {
    val zone = ZoneId.of( "GMT+3" )
    val date = LocalDate.now(zone).`with`(TemporalAdjusters.previousOrSame( DayOfWeek.MONDAY ))
    date.atStartOfDay(zone).toEpochSecond
  }

  private val gmtId = ZoneId.of("GMT")
  private val formatterYMD = DateTimeFormatter.ofPattern("yyyy-MM-dd", Locale.ENGLISH)
  private val formatterYMDHMS = DateTimeFormatter.ofPattern("yyyy-MM-dd HH-mm-ss", Locale.ENGLISH)

  def getYMD(): String = LocalDateTime.now(gmtId).format(formatterYMD)
  def getYMDHMS(): String = LocalDateTime.now(gmtId).format(formatterYMDHMS)

  implicit class MyStringExt(val str: String) extends AnyVal {
    def -(suffix: String): String = {
      require(str.endsWith(suffix))
      str.substring(0, str.size - suffix.size)
    }

    def quoted: String = s"\"$str\""

    def quoted(maxLength: Int, suffix: String = "..."): String =
      limitLength(maxLength, suffix).quoted

    def limitLength(maxLength: Int, suffix: String = "..."): String = {
      require(maxLength >= suffix.size)
      if (str.size > maxLength)
        str.substring(0, maxLength - suffix.size) + suffix
      else
        str
    }
  }

  implicit class MyFileExt(val file: File) extends AnyVal {
    private def charset = "utf-8"

    def text: String = {
      val source = scala.io.Source.fromFile(file, charset)
      val text = source.getLines().mkString("\n")
      source.close()
      text
    }

    def lines: Seq[String] = {
      val source = scala.io.Source.fromFile(file, charset)
      val lines = source.getLines().toSeq
      source.close()
      lines
    }

    def text_=(newText: String): Unit = {
      Files.write(file.toPath, newText.getBytes(charset),
        StandardOpenOption.WRITE, StandardOpenOption.CREATE)
    }

    def <<(appendText: String): Unit = {
      Files.write(file.toPath, appendText.getBytes(charset),
        StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    }
  }

  implicit class MyIntExt(val v: Int) extends AnyVal {
    def clamp(lower: Int, upper: Int): Int =
      math.min(upper, math.max(lower, v))
  }
}
