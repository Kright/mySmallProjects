package me.kright.raytracer

import java.util.concurrent.TimeUnit
import me.kright.arrayview.ArrayView3d

import java.io.FileWriter
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

@main
def main(): Unit =
  val runName: String = "batch_bvh_complex_scene"

  val imageWidth = 1200
  val imageHeight = 675
  val aspectRatio = imageWidth.toDouble / imageHeight.toDouble
  val samplesPerPixel = 50
  val maxDepth = 50

  val worldList = Scene.generateRandomScene()
  println(s"Objects in world: ${worldList.objects.length}")
  val world = BVHNode(worldList)

  val camera = Camera(
    Vector3(13, 2, 3),
    Vector3(0, 0, 0),
    Vector3(0, 1, 0),
    20,
    aspectRatio
  )

  val accumulator = ArrayView3d[Double](imageHeight, imageWidth, 3)

  val start = System.nanoTime()
  Renderer.renderParallel(accumulator, imageWidth, imageHeight, samplesPerPixel, maxDepth, camera, world)
  val elapsedNanos = System.nanoTime() - start
  val elapsedSecs = elapsedNanos / 1e9
  println(f"Render time: $elapsedSecs%.3f s")

  val timestamp = LocalDateTime.now.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
  val fw = FileWriter("benchmarks.tsv", true)
  try fw.write(s"$runName\t$elapsedNanos\t$timestamp\n")
  finally fw.close()

  saveAsPng(accumulator, "output.png", 2.2)

