package me.kright.raytracer

case class Camera(
  lookFrom: Vector3,
  lookAt: Vector3,
  vup: Vector3,
  vfov: Double,
  aspectRatio: Double
):
  private val theta = math.toRadians(vfov)
  private val h = math.tan(theta / 2)
  private val viewportHeight = 2 * h
  private val viewportWidth = aspectRatio * viewportHeight

  private val w = (lookFrom - lookAt).normalized
  private val u = vup.cross(w).normalized
  private val v = w.cross(u)

  private val origin = lookFrom
  private val horizontal = u * viewportWidth
  private val vertical = v * viewportHeight
  private val lowerLeftCorner = origin - horizontal / 2 - vertical / 2 - w

  def getRay(s: Double, t: Double): Ray =
    Ray(origin, lowerLeftCorner + horizontal * s + vertical * t - origin)

  def initRayInBatch(batch: RayBatch, i: Int, s: Double, t: Double, px: Int, py: Int): Unit =
    batch.origins(i, 0) = origin.x
    batch.origins(i, 1) = origin.y
    batch.origins(i, 2) = origin.z
    batch.directions(i, 0) = lowerLeftCorner.x + horizontal.x * s + vertical.x * t - origin.x
    batch.directions(i, 1) = lowerLeftCorner.y + horizontal.y * s + vertical.y * t - origin.y
    batch.directions(i, 2) = lowerLeftCorner.z + horizontal.z * s + vertical.z * t - origin.z
    batch.attenuations(i, 0) = 1.0
    batch.attenuations(i, 1) = 1.0
    batch.attenuations(i, 2) = 1.0
    batch.colors(i, 0) = 0.0
    batch.colors(i, 1) = 0.0
    batch.colors(i, 2) = 0.0
    batch.active(i) = true
    batch.pixelX(i) = px
    batch.pixelY(i) = py
