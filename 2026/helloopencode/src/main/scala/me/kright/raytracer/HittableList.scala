package me.kright.raytracer

import scala.IArray

case class HittableList(objects: IArray[Hittable]) extends Hittable:
  def hit(ray: Ray, tMin: Double, tMax: Double): HitRecord | Null =
    var closestT = tMax
    var result: HitRecord | Null = null
    var i = 0
    while i < objects.length do
      val rec = objects(i).hit(ray, tMin, closestT)
      if rec != null then
        result = rec
        closestT = rec.t
      i += 1
    result
  
  def hitBatch(batch: RayBatch, count: Int, tMin: Double): Unit =
    var i = 0
    while i < objects.length do
      objects(i).hitBatch(batch, count, tMin)
      i += 1

  override def boundingBox: AABB =
    if objects.isEmpty then AABB(0, 0, 0, 0, 0, 0)
    else
      var tempBox = objects(0).boundingBox
      var i = 1
      while i < objects.length do
        tempBox = AABB.surroundingBox(tempBox, objects(i).boundingBox)
        i += 1
      tempBox
