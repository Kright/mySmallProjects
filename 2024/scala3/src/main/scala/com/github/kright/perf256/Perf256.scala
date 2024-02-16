package com.github.kright.perf256

import scala.util.Random

@main
def main(): Unit =
  val screen = Screen(3840, 2160)
  val palettedScreen = PalettedScreen(3840, 2160)

  val rnd = Random(0xdeadbeef)

  def makeRandomPic(w: Int, h: Int): CompressedPixelPicture =
    CompressedPixelPicture((0 until h).map(i => (rnd.nextInt(w/2).toByte, rnd.nextBytes(rnd.nextInt(w/2)))).toArray)

  val pics = (0 until 1000).map(_ => makeRandomPic(40, 50))
  val largePics = (0 until 1000).map(_ => makeRandomPic(400, 500))

  for (i <- 0 until 10000) {
    val pic = pics(rnd.nextInt(pics.size))
    val x = rnd.nextInt(screen.w - pic.w)
    val y = rnd.nextInt(screen.h - pic.h)
    pic.drawFasta(screen, x, y)
  }

  printTimeMs("draw 10000 pics fasta") {
    for (i <- 0 until 10000) {
      val pic = pics(rnd.nextInt(pics.size))
      val x = rnd.nextInt(screen.w - pic.w)
      val y = rnd.nextInt(screen.h - pic.h)
      pic.drawFasta(screen, x, y)
    }
  }

  for (i <- 0 until 10000) {
    val pic = pics(rnd.nextInt(pics.size))
    val x = rnd.nextInt(screen.w - pic.w)
    val y = rnd.nextInt(screen.h - pic.h)
    pic.draw(screen, x, y)
  }

  printTimeMs("draw 10000 pics") {
    for (i <- 0 until 10000) {
      val pic = pics(rnd.nextInt(pics.size))
      val x = rnd.nextInt(screen.w - pic.w)
      val y = rnd.nextInt(screen.h - pic.h)
      pic.draw(screen, x, y)
    }
  }

  for (i <- 0 until 1000) {
    val pic = largePics(rnd.nextInt(largePics.size))
    val x = rnd.nextInt(screen.w - pic.w)
    val y = rnd.nextInt(screen.h - pic.h)
    pic.draw(screen, x, y)
  }

  printTimeMs("draw 1000 large pics") {
    for (i <- 0 until 1000) {
      val pic = largePics(rnd.nextInt(largePics.size))
      val x = rnd.nextInt(screen.w - pic.w)
      val y = rnd.nextInt(screen.h - pic.h)
      pic.draw(screen, x, y)
    }
  }

  println(s"pixel rate = ${pics.map(_.data.length).sum * 10000L / pics.size}")

  for (i <- 0 until 10) {
    printTimeMs("apply palette") {
      palettedScreen.setFromScreen(screen)
    }
  }


private def printTimeMs(msg: String)(f: => Unit): Unit =
  val t0 = System.nanoTime()
  f
  val t1 = System.nanoTime()
  println(s"$msg ${t1 - t0}")

class Screen(val w: Int, val h: Int):
  val pixels: Array[Byte] = new Array(w * h)

class PalettedScreen(val w: Int, val h: Int) {
  val palette: Array[Int] = (0 until 256).map(i => 0xFF000000 | (i * 0x10101)).toArray
  val pixels: Array[Int] = new Array[Int](w * h)

  def setFromScreen(screen: Screen): Unit = {
    val m = w * h
    var i = 0
    val sc = screen.pixels
    val p = palette
    val pix = pixels
    while( i < m) {
      pix(i) = palette(sc(i).toInt & 0x000000FF)
      i += 1
    }
  }
}

class CompressedPixelPicture(val w: Int,
                             val h: Int,
                             val commands: Array[Byte],
                             val data: Array[Byte]):

  def draw(screen: Screen, x: Int, y: Int): Unit =
    if (!isFitsInScreen(screen, x, y)) return

    var dataPos = 0
    var cmdPos = 0
    val pixels = screen.pixels
    val sw = screen.w

    val ymax = h + y
    var localY = y
    while (localY < ymax) {
      localY += 1

      val offset = commands(cmdPos)
      val count = commands(cmdPos + 1)
      cmdPos += 2

      val startPos = sw * localY + x + offset

      var i = 0
      while (i < count) {
        pixels(startPos + i) = data(dataPos)
        dataPos += 1
        i += 1
      }
    }


  def drawFasta(screen: Screen, x: Int, y: Int): Unit =
    if (!isFitsInScreen(screen, x, y)) return

    var dataPos = 0
    var cmdPos = 0
    val pixels = screen.pixels
    val sw = screen.w

    val ymax = h + y
    var localY = y
    while (localY < ymax){
      localY +=1

      val offset = commands(cmdPos)
      val count = commands(cmdPos + 1)
      cmdPos += 2

      val startPos = sw * localY + x + offset
      System.arraycopy(data, dataPos, pixels, startPos, count)
      dataPos += count
    }

  def isFitsInScreen(screen: Screen, x: Int, y: Int): Boolean =
    x >= 0 && y >=0 && x + w < screen.w && y + h < screen.h

object CompressedPixelPicture:
  def apply(lines: Array[(Byte, Array[Byte])]): CompressedPixelPicture =
    val w = lines.map((offset, line) => offset + line.length).max.toByte
    val h = lines.size
    val commands = lines.flatMap((offset, line) => Array(offset.toByte, line.length.toByte))
    val data = lines.flatMap((offset, line) => line)
    new CompressedPixelPicture(w, h, commands, data)
