package me.kright.raytracer

case class AABB(minX: Double, minY: Double, minZ: Double, maxX: Double, maxY: Double, maxZ: Double):
  def hit(ray: Ray, tMin: Double, tMax: Double): Boolean =
    var t0 = tMin
    var t1 = tMax
    
    val ox = ray.origin.x
    val oy = ray.origin.y
    val oz = ray.origin.z
    val dx = ray.direction.x
    val dy = ray.direction.y
    val dz = ray.direction.z

    // X
    val invDx = 1.0 / dx
    var tNear = (minX - ox) * invDx
    var tFar = (maxX - ox) * invDx
    if invDx < 0 then
      val tmp = tNear
      tNear = tFar
      tFar = tmp
    t0 = Math.max(t0, tNear)
    t1 = Math.min(t1, tFar)
    if t1 <= t0 then return false

    // Y
    val invDy = 1.0 / dy
    tNear = (minY - oy) * invDy
    tFar = (maxY - oy) * invDy
    if invDy < 0 then
      val tmp = tNear
      tNear = tFar
      tFar = tmp
    t0 = Math.max(t0, tNear)
    t1 = Math.min(t1, tFar)
    if t1 <= t0 then return false

    // Z
    val invDz = 1.0 / dz
    tNear = (minZ - oz) * invDz
    tFar = (maxZ - oz) * invDz
    if invDz < 0 then
      val tmp = tNear
      tNear = tFar
      tFar = tmp
    t0 = Math.max(t0, tNear)
    t1 = Math.min(t1, tFar)
    if t1 <= t0 then return false
    
    true

  def hitInBatch(batch: RayBatch, i: Int, tMin: Double, tMax: Double): Boolean =
    var t0 = tMin
    var t1 = tMax
    
    val ox = batch.origins(i, 0)
    val oy = batch.origins(i, 1)
    val oz = batch.origins(i, 2)
    val dx = batch.directions(i, 0)
    val dy = batch.directions(i, 1)
    val dz = batch.directions(i, 2)

    // X
    val invDx = 1.0 / dx
    var tNear = (minX - ox) * invDx
    var tFar = (maxX - ox) * invDx
    if invDx < 0 then
      val tmp = tNear
      tNear = tFar
      tFar = tmp
    t0 = Math.max(t0, tNear)
    t1 = Math.min(t1, tFar)
    if t1 <= t0 then return false

    // Y
    val invDy = 1.0 / dy
    tNear = (minY - oy) * invDy
    tFar = (maxY - oy) * invDy
    if invDy < 0 then
      val tmp = tNear
      tNear = tFar
      tFar = tmp
    t0 = Math.max(t0, tNear)
    t1 = Math.min(t1, tFar)
    if t1 <= t0 then return false

    // Z
    val invDz = 1.0 / dz
    tNear = (minZ - oz) * invDz
    tFar = (maxZ - oz) * invDz
    if invDz < 0 then
      val tmp = tNear
      tNear = tFar
      tFar = tmp
    t0 = Math.max(t0, tNear)
    t1 = Math.min(t1, tFar)
    if t1 <= t0 then return false
    
    true

object AABB:
  def surroundingBox(box0: AABB, box1: AABB): AABB =
    AABB(
      Math.min(box0.minX, box1.minX),
      Math.min(box0.minY, box1.minY),
      Math.min(box0.minZ, box1.minZ),
      Math.max(box0.maxX, box1.maxX),
      Math.max(box0.maxY, box1.maxY),
      Math.max(box0.maxZ, box1.maxZ)
    )
