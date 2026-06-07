package me.kright.raytracer

case class Sphere(center: Vector3, radius: Double, material: Material) extends Hittable:
  val invRadius = 1.0 / radius
  val radiusSq = radius * radius

  def hit(ray: Ray, tMin: Double, tMax: Double): HitRecord | Null =
    val ocX = ray.origin.x - center.x
    val ocY = ray.origin.y - center.y
    val ocZ = ray.origin.z - center.z
    
    val dx = ray.direction.x
    val dy = ray.direction.y
    val dz = ray.direction.z

    val a = Math.fma(dx, dx, Math.fma(dy, dy, dz * dz))
    val halfB = Math.fma(ocX, dx, Math.fma(ocY, dy, ocZ * dz))
    val c = Math.fma(ocX, ocX, Math.fma(ocY, ocY, Math.fma(ocZ, ocZ, -radiusSq)))
    val discriminant = Math.fma(halfB, halfB, -a * c)

    if discriminant < 0 then null
    else
      val sqrtD = math.sqrt(discriminant)
      val invA = 1.0 / a
      var root = (-halfB - sqrtD) * invA
      if root < tMin || tMax < root then
        root = (-halfB + sqrtD) * invA
        if root < tMin || tMax < root then
          return null

      val px = Math.fma(dx, root, ray.origin.x)
      val py = Math.fma(dy, root, ray.origin.y)
      val pz = Math.fma(dz, root, ray.origin.z)
      val point = Vector3(px, py, pz)

      val outNormX = (px - center.x) * invRadius
      val outNormY = (py - center.y) * invRadius
      val outNormZ = (pz - center.z) * invRadius
      
      val frontFace = Math.fma(dx, outNormX, Math.fma(dy, outNormY, dz * outNormZ)) < 0
      val normal = if frontFace then Vector3(outNormX, outNormY, outNormZ) else Vector3(-outNormX, -outNormY, -outNormZ)
      HitRecord(point, normal, root, frontFace, material)

  def hitBatch(batch: RayBatch, count: Int, tMin: Double): Unit =
    val cx = center.x
    val cy = center.y
    val cz = center.z

    var i = 0
    while i < count do
      if batch.active(i) then
        val dx = batch.directions(i, 0)
        val dy = batch.directions(i, 1)
        val dz = batch.directions(i, 2)
        
        val ox = batch.origins(i, 0) - cx
        val oy = batch.origins(i, 1) - cy
        val oz = batch.origins(i, 2) - cz

        val a = Math.fma(dx, dx, Math.fma(dy, dy, dz * dz))
        val halfB = Math.fma(ox, dx, Math.fma(oy, dy, oz * dz))
        val c = Math.fma(ox, ox, Math.fma(oy, oy, Math.fma(oz, oz, -radiusSq)))
        val discriminant = Math.fma(halfB, halfB, -a * c)

        if discriminant >= 0 then
          val sqrtD = math.sqrt(discriminant)
          val invA = 1.0 / a
          var root = (-halfB - sqrtD) * invA
          if root < tMin || batch.hitT(i) < root then
            root = (-halfB + sqrtD) * invA
          
          if tMin <= root && root < batch.hitT(i) then
            batch.hitT(i) = root
            batch.hitMaterial(i) = material
            
            val px = Math.fma(dx, root, batch.origins(i, 0))
            val py = Math.fma(dy, root, batch.origins(i, 1))
            val pz = Math.fma(dz, root, batch.origins(i, 2))
            batch.hitPoints(i, 0) = px
            batch.hitPoints(i, 1) = py
            batch.hitPoints(i, 2) = pz
            
            val outNormX = (px - cx) * invRadius
            val outNormY = (py - cy) * invRadius
            val outNormZ = (pz - cz) * invRadius
            
            val frontFace = Math.fma(dx, outNormX, Math.fma(dy, outNormY, dz * outNormZ)) < 0
            batch.hitFrontFace(i) = frontFace
            if frontFace then
              batch.hitNormals(i, 0) = outNormX
              batch.hitNormals(i, 1) = outNormY
              batch.hitNormals(i, 2) = outNormZ
            else
              batch.hitNormals(i, 0) = -outNormX
              batch.hitNormals(i, 1) = -outNormY
              batch.hitNormals(i, 2) = -outNormZ
      i += 1

  override def boundingBox: AABB =
    val r = math.abs(radius)
    AABB(
      center.x - r, center.y - r, center.z - r,
      center.x + r, center.y + r, center.z + r
    )
