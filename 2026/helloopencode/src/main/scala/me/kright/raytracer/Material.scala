package me.kright.raytracer

import java.util.concurrent.ThreadLocalRandom

trait Material:
  def scatter(ray: Ray, rec: HitRecord): (Ray, Color) | Null

  def scatterBatch(batch: RayBatch, i: Int): Boolean

case class Lambertian(albedo: Color) extends Material:
  private val attenuation = albedo
  def scatter(ray: Ray, rec: HitRecord): (Ray, Color) | Null =
    val scatterDirection = rec.normal + Vector3.randomUnitVector()
    val scattered =
      if scatterDirection.nearZero then Ray(rec.point, rec.normal)
      else Ray(rec.point, scatterDirection)
    (scattered, attenuation)

  def scatterBatch(batch: RayBatch, i: Int): Boolean =
    val rnd = ThreadLocalRandom.current()
    val z = rnd.nextDouble(-1, 1)
    val a = rnd.nextDouble(0, 2 * math.Pi)
    val r = math.sqrt(1.0 - z * z)
    val rvx = r * math.cos(a)
    val rvy = r * math.sin(a)
    val rvz = z

    val nx = batch.hitNormals(i, 0)
    val ny = batch.hitNormals(i, 1)
    val nz = batch.hitNormals(i, 2)
    
    var sx = nx + rvx
    var sy = ny + rvy
    var sz = nz + rvz
    
    if sx * sx + sy * sy + sz * sz < 1e-16 then
      sx = nx
      sy = ny
      sz = nz
    
    batch.origins(i, 0) = batch.hitPoints(i, 0)
    batch.origins(i, 1) = batch.hitPoints(i, 1)
    batch.origins(i, 2) = batch.hitPoints(i, 2)
    batch.directions(i, 0) = sx
    batch.directions(i, 1) = sy
    batch.directions(i, 2) = sz
    batch.attenuations(i, 0) *= albedo.r
    batch.attenuations(i, 1) *= albedo.g
    batch.attenuations(i, 2) *= albedo.b
    true

case class Metal(albedo: Color, fuzz: Double) extends Material:
  def scatter(ray: Ray, rec: HitRecord): (Ray, Color) | Null =
    val reflected = reflect(ray.direction.normalized, rec.normal)
    val scattered = Ray(rec.point, reflected + Vector3.randomInUnitSphere() * fuzz)
    if scattered.direction.dot(rec.normal) > 0 then (scattered, albedo)
    else null

  def scatterBatch(batch: RayBatch, i: Int): Boolean =
    val dx = batch.directions(i, 0)
    val dy = batch.directions(i, 1)
    val dz = batch.directions(i, 2)
    val nx = batch.hitNormals(i, 0)
    val ny = batch.hitNormals(i, 1)
    val nz = batch.hitNormals(i, 2)
    
    val invLen = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
    val udx = dx * invLen
    val udy = dy * invLen
    val udz = dz * invLen
    
    val dot = udx * nx + udy * ny + udz * nz
    var rx = udx - 2 * dot * nx
    var ry = udy - 2 * dot * ny
    var rz = udz - 2 * dot * nz
    
    if fuzz > 0 then
      val rnd = ThreadLocalRandom.current()
      var rvx, rvy, rvz = 0.0
      while {
        rvx = rnd.nextDouble(-1, 1)
        rvy = rnd.nextDouble(-1, 1)
        rvz = rnd.nextDouble(-1, 1)
        rvx * rvx + rvy * rvy + rvz * rvz >= 1.0
      } do ()
      rx += fuzz * rvx
      ry += fuzz * rvy
      rz += fuzz * rvz
      
    if rx * nx + ry * ny + rz * nz > 0 then
      batch.origins(i, 0) = batch.hitPoints(i, 0)
      batch.origins(i, 1) = batch.hitPoints(i, 1)
      batch.origins(i, 2) = batch.hitPoints(i, 2)
      batch.directions(i, 0) = rx
      batch.directions(i, 1) = ry
      batch.directions(i, 2) = rz
      batch.attenuations(i, 0) *= albedo.r
      batch.attenuations(i, 1) *= albedo.g
      batch.attenuations(i, 2) *= albedo.b
      true
    else
      false

  private def reflect(v: Vector3, n: Vector3): Vector3 = v - n * 2 * v.dot(n)

case class Dielectric(refractiveIndex: Double) extends Material:
  def scatter(ray: Ray, rec: HitRecord): (Ray, Color) | Null =
    val attenuation = Color(1, 1, 1)
    val refractionRatio = if rec.frontFace then 1.0 / refractiveIndex else refractiveIndex

    val unitDirection = ray.direction.normalized
    val cosTheta = (-unitDirection.dot(rec.normal)).min(1.0)
    val sinTheta = math.sqrt(1 - cosTheta * cosTheta)

    val cannotRefract = refractionRatio * sinTheta > 1
    val rnd = ThreadLocalRandom.current()
    val direction =
      if cannotRefract || reflectance(cosTheta, refractionRatio) > rnd.nextDouble() then
        reflect(unitDirection, rec.normal)
      else
        refract(unitDirection, rec.normal, refractionRatio)

    (Ray(rec.point, direction), attenuation)

  def scatterBatch(batch: RayBatch, i: Int): Boolean =
    val refractionRatio = if batch.hitFrontFace(i) then 1.0 / refractiveIndex else refractiveIndex
    
    val dx = batch.directions(i, 0)
    val dy = batch.directions(i, 1)
    val dz = batch.directions(i, 2)
    val nx = batch.hitNormals(i, 0)
    val ny = batch.hitNormals(i, 1)
    val nz = batch.hitNormals(i, 2)
    
    val invLen = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
    val udx = dx * invLen
    val udy = dy * invLen
    val udz = dz * invLen
    
    val cosTheta = (-udx * nx - udy * ny - udz * nz).min(1.0)
    val sinTheta = math.sqrt(1.0 - cosTheta * cosTheta)
    
    val cannotRefract = refractionRatio * sinTheta > 1.0
    val rnd = ThreadLocalRandom.current()
    
    var resX, resY, resZ = 0.0
    
    if cannotRefract || reflectance(cosTheta, refractionRatio) > rnd.nextDouble() then
      // reflect
      resX = udx + 2 * cosTheta * nx
      resY = udy + 2 * cosTheta * ny
      resZ = udz + 2 * cosTheta * nz
    else
      // refract
      val rOutParallelX = (udx + nx * cosTheta) * refractionRatio
      val rOutParallelY = (udy + ny * cosTheta) * refractionRatio
      val rOutParallelZ = (udz + nz * cosTheta) * refractionRatio
      val rOutParallelSqLen = rOutParallelX * rOutParallelX + rOutParallelY * rOutParallelY + rOutParallelZ * rOutParallelZ
      val rOutPerpFactor = -math.sqrt(math.abs(1.0 - rOutParallelSqLen))
      resX = rOutParallelX + nx * rOutPerpFactor
      resY = rOutParallelY + ny * rOutPerpFactor
      resZ = rOutParallelZ + nz * rOutPerpFactor
      
    batch.origins(i, 0) = batch.hitPoints(i, 0)
    batch.origins(i, 1) = batch.hitPoints(i, 1)
    batch.origins(i, 2) = batch.hitPoints(i, 2)
    batch.directions(i, 0) = resX
    batch.directions(i, 1) = resY
    batch.directions(i, 2) = resZ
    true

  private def reflect(v: Vector3, n: Vector3): Vector3 = v - n * 2 * v.dot(n)

  private def refract(uv: Vector3, n: Vector3, etaiOverEtat: Double): Vector3 =
    val cosTheta = -uv.dot(n)
    val rOutParallel = (uv + n * cosTheta) * etaiOverEtat
    val rOutPerp = n * (-math.sqrt(math.abs(1 - rOutParallel.squaredLength)))
    rOutParallel + rOutPerp

  private def reflectance(cosine: Double, refIdx: Double): Double =
    val r0 = (1 - refIdx) / (1 + refIdx)
    val r0Sq = r0 * r0
    val a = 1 - cosine
    r0Sq + (1 - r0Sq) * a * ((a * a) * (a * a)) 
