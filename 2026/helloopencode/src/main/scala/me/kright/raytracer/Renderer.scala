package me.kright.raytracer

import java.util.concurrent.{Executors, ThreadLocalRandom, TimeUnit}
import me.kright.arrayview.ArrayView3d

object Renderer:
  def renderParallel(
    acc: ArrayView3d[Double],
    imageWidth: Int,
    imageHeight: Int,
    samplesPerPixel: Int,
    maxDepth: Int,
    camera: Camera,
    world: Hittable
  ): Unit =
    val tileSize = 128
    val numThreads = Runtime.getRuntime.availableProcessors()
    val pool = Executors.newFixedThreadPool(numThreads)
    val futures =
      (for
        yStart <- (0 until imageHeight by tileSize)
        xStart <- (0 until imageWidth by tileSize)
      yield
        val yEnd = (yStart + tileSize).min(imageHeight)
        val xEnd = (xStart + tileSize).min(imageWidth)
        pool.submit[Unit] { () => accumulateTile(acc, xStart, xEnd, yStart, yEnd, imageWidth, imageHeight, samplesPerPixel, maxDepth, camera, world); () }
      ).toVector
    for f <- futures do f.get()
    pool.shutdown()
    pool.awaitTermination(1, TimeUnit.DAYS)
    ()

  private def accumulateTile(
    acc: ArrayView3d[Double],
    xStart: Int, xEnd: Int,
    yStart: Int, yEnd: Int,
    imageWidth: Int,
    imageHeight: Int,
    samplesPerPixel: Int,
    maxDepth: Int,
    camera: Camera,
    world: Hittable
  ): Unit = {
    val batchSize = 1024
    val batch = new RayBatch(batchSize)
    val rnd = ThreadLocalRandom.current()

    var currentX = xStart
    var currentY = yStart
    var currentS = 0

    while currentY < yEnd do
      var count = 0
      while count < batchSize && currentY < yEnd do
        val u = (currentX + rnd.nextDouble()) / (imageWidth - 1)
        val v = (currentY + rnd.nextDouble()) / (imageHeight - 1)
        camera.initRayInBatch(batch, count, u, v, currentX, currentY)
        
        count += 1
        currentS += 1
        if currentS >= samplesPerPixel then
          currentS = 0
          currentX += 1
          if currentX >= xEnd then
            currentX = xStart
            currentY += 1
      
      var activeCount = count
      var d = 0
      while d < maxDepth && activeCount > 0 do
        batch.resetHits(activeCount)
        world.hitBatch(batch, activeCount, 0.001)
        
        var i = 0
        var nextActiveCount = 0
        while i < activeCount do
          val mat = batch.hitMaterial(i)
          if mat != null then
            if mat.scatterBatch(batch, i) then
              batch.swap(nextActiveCount, i)
              nextActiveCount += 1
          else
            val dx = batch.directions(i, 0)
            val dy = batch.directions(i, 1)
            val dz = batch.directions(i, 2)
            val invLen = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
            val t = 0.5 * (dy * invLen + 1.0)
            val r = (1.0 - t) + 0.5 * t
            val g = (1.0 - t) + 0.7 * t
            val b = (1.0 - t) + 1.0 * t
            
            batch.colors(nextActiveCount, 0) = batch.colors(i, 0) + batch.attenuations(i, 0) * r
            batch.colors(nextActiveCount, 1) = batch.colors(i, 1) + batch.attenuations(i, 1) * g
            batch.colors(nextActiveCount, 2) = batch.colors(i, 2) + batch.attenuations(i, 2) * b
            batch.pixelX(nextActiveCount) = batch.pixelX(i)
            batch.pixelY(nextActiveCount) = batch.pixelY(i)
            
            val row = imageHeight - 1 - batch.pixelY(nextActiveCount)
            val col = batch.pixelX(nextActiveCount)
            val invSamples = 1.0 / samplesPerPixel
            acc(row, col, 0) += batch.colors(nextActiveCount, 0) * invSamples
            acc(row, col, 1) += batch.colors(nextActiveCount, 1) * invSamples
            acc(row, col, 2) += batch.colors(nextActiveCount, 2) * invSamples
          i += 1
        activeCount = nextActiveCount
        d += 1
  }
