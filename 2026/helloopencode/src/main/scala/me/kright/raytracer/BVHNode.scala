package me.kright.raytracer

import scala.util.Random

class BVHNode(val left: Hittable, val right: Hittable, val box: AABB) extends Hittable:

  override def hit(ray: Ray, tMin: Double, tMax: Double): HitRecord | Null =
    if !box.hit(ray, tMin, tMax) then return null

    val hitLeft = left.hit(ray, tMin, tMax)
    val hitRight = right.hit(ray, tMin, if hitLeft != null then hitLeft.t else tMax)

    if hitRight != null then hitRight else hitLeft

  override def hitBatch(batch: RayBatch, count: Int, tMin: Double): Unit =
    var anyHit = false
    var i = 0
    while i < count do
      if batch.active(i) && box.hitInBatch(batch, i, tMin, batch.hitT(i)) then
        anyHit = true
        i = count
      else
        i += 1
    
    if anyHit then
      left.hitBatch(batch, count, tMin)
      right.hitBatch(batch, count, tMin)

  override def boundingBox: AABB = box

object BVHNode:
  def apply(list: HittableList): BVHNode | Hittable =
    val objs = list.objects.toArray
    apply(objs, 0, objs.length)

  def apply(objects: Array[Hittable], start: Int, end: Int): BVHNode | Hittable =
    val axis = Random.nextInt(3)
    
    val objectSpan = end - start
    if objectSpan == 1 then
      return objects(start)
    
    if objectSpan == 2 then
      val left = objects(start)
      val right = objects(start + 1)
      val box = AABB.surroundingBox(left.boundingBox, right.boundingBox)
      return new BVHNode(left, right, box)

    // Sort
    java.util.Arrays.sort(objects, start, end, (a, b) => {
      val boxA = a.boundingBox
      val boxB = b.boundingBox
      val res = axis match
        case 0 => boxA.minX < boxB.minX
        case 1 => boxA.minY < boxB.minY
        case 2 => boxA.minZ < boxB.minZ
      if res then -1 else 1
    })
    
    val mid = start + objectSpan / 2
    val left = apply(objects, start, mid)
    val right = apply(objects, mid, end)
    val box = AABB.surroundingBox(left.boundingBox, right.boundingBox)
    new BVHNode(left, right, box)
