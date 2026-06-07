package me.kright.raytracer

import me.kright.arrayview.ArrayView2d

class RayBatch(val capacity: Int):
  val origins = ArrayView2d[Double](capacity, 3)
  val directions = ArrayView2d[Double](capacity, 3)
  val attenuations = ArrayView2d[Double](capacity, 3)
  val colors = ArrayView2d[Double](capacity, 3)
  
  val active = new Array[Boolean](capacity)
  val depth = new Array[Int](capacity)
  
  val pixelX = new Array[Int](capacity)
  val pixelY = new Array[Int](capacity)

  val hitT = new Array[Double](capacity)
  val hitNormals = ArrayView2d[Double](capacity, 3)
  val hitPoints = ArrayView2d[Double](capacity, 3)
  val hitFrontFace = new Array[Boolean](capacity)
  val hitMaterial = new Array[Material | Null](capacity)

  def initRay(i: Int, ox: Double, oy: Double, oz: Double, dx: Double, dy: Double, dz: Double, px: Int, py: Int): Unit =
    origins(i, 0) = ox
    origins(i, 1) = oy
    origins(i, 2) = oz
    directions(i, 0) = dx
    directions(i, 1) = dy
    directions(i, 2) = dz
    attenuations(i, 0) = 1.0
    attenuations(i, 1) = 1.0
    attenuations(i, 2) = 1.0
    colors(i, 0) = 0.0
    colors(i, 1) = 0.0
    colors(i, 2) = 0.0
    active(i) = true
    depth(i) = 0
    pixelX(i) = px
    pixelY(i) = py

  def resetHits(count: Int): Unit =
    var i = 0
    while i < count do
      hitT(i) = Double.MaxValue
      hitMaterial(i) = null
      i += 1

  def swap(i: Int, j: Int): Unit =
    if i == j then return
    
    var c = 0
    while c < 3 do
      val ox = origins(i, c)
      origins(i, c) = origins(j, c)
      origins(j, c) = ox

      val dx = directions(i, c)
      directions(i, c) = directions(j, c)
      directions(j, c) = dx

      val ax = attenuations(i, c)
      attenuations(i, c) = attenuations(j, c)
      attenuations(j, c) = ax

      val cx = colors(i, c)
      colors(i, c) = colors(j, c)
      colors(j, c) = cx
      c += 1

    val px = pixelX(i)
    pixelX(i) = pixelX(j)
    pixelX(j) = px

    val py = pixelY(i)
    pixelY(i) = pixelY(j)
    pixelY(j) = py

    val ht = hitT(i)
    hitT(i) = hitT(j)
    hitT(j) = ht

    val hm = hitMaterial(i)
    hitMaterial(i) = hitMaterial(j)
    hitMaterial(j) = hm

    val hff = hitFrontFace(i)
    hitFrontFace(i) = hitFrontFace(j)
    hitFrontFace(j) = hff

    val hnx = hitNormals(i, 0)
    val hny = hitNormals(i, 1)
    val hnz = hitNormals(i, 2)
    hitNormals(i, 0) = hitNormals(j, 0)
    hitNormals(i, 1) = hitNormals(j, 1)
    hitNormals(i, 2) = hitNormals(j, 2)
    hitNormals(j, 0) = hnx
    hitNormals(j, 1) = hny
    hitNormals(j, 2) = hnz

    val hpx = hitPoints(i, 0)
    val hpy = hitPoints(i, 1)
    val hpz = hitPoints(i, 2)
    hitPoints(i, 0) = hitPoints(j, 0)
    hitPoints(i, 1) = hitPoints(j, 1)
    hitPoints(i, 2) = hitPoints(j, 2)
    hitPoints(j, 0) = hpx
    hitPoints(j, 1) = hpy
    hitPoints(j, 2) = hpz
