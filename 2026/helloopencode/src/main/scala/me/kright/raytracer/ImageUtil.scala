package me.kright.raytracer

import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File
import me.kright.arrayview.ArrayView3d

def saveAsPng(image: ArrayView3d[Double], filename: String, gamma: Double): Unit =
  val imageHeight = image.shape0
  val imageWidth = image.shape1
  val img = BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_RGB)
  val invGamma = 1.0 / gamma
  for (y <- 0 until imageHeight; x <- 0 until imageWidth)
    val r = image(y, x, 0)
    val g = image(y, x, 1)
    val b = image(y, x, 2)
    def correct(v: Double): Int =
      (math.pow(v.max(0.0).min(0.999), invGamma) * 255.999).toInt
    img.setRGB(x, y, (correct(r) << 16) | (correct(g) << 8) | correct(b))
  ImageIO.write(img, "png", File(filename))
