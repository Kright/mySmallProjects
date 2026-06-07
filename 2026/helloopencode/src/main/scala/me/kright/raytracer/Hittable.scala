package me.kright.raytracer

case class HitRecord(point: Vector3, normal: Vector3, t: Double, frontFace: Boolean, material: Material)

trait Hittable:
  def hit(ray: Ray, tMin: Double, tMax: Double): HitRecord | Null

  def hitBatch(batch: RayBatch, count: Int, tMin: Double): Unit

  def boundingBox: AABB
