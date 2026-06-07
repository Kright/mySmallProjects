package me.kright.raytracer

import java.util.concurrent.ThreadLocalRandom

case class Vector3(x: Double, y: Double, z: Double):
  inline def +(v: Vector3): Vector3 = Vector3(x + v.x, y + v.y, z + v.z)
  inline def -(v: Vector3): Vector3 = Vector3(x - v.x, y - v.y, z - v.z)
  inline def *(t: Double): Vector3 = Vector3(x * t, y * t, z * t)
  inline def /(t: Double): Vector3 = this * (1 / t)
  inline def dot(v: Vector3): Double = x * v.x + y * v.y + z * v.z
  inline def cross(v: Vector3): Vector3 = Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x)
  inline def length: Double = Math.sqrt(x * x + y * y + z * z)
  inline def squaredLength: Double = x * x + y * y + z * z
  inline def normalized: Vector3 = this / length
  inline def negate: Vector3 = Vector3(-x, -y, -z)
  inline def nearZero: Boolean = math.abs(x) < 1e-8 && math.abs(y) < 1e-8 && math.abs(z) < 1e-8

object Vector3:
  def random(min: Double = 0, max: Double = 1): Vector3 =
    Vector3(
      ThreadLocalRandom.current().nextDouble(min, max),
      ThreadLocalRandom.current().nextDouble(min, max),
      ThreadLocalRandom.current().nextDouble(min, max)
    )

  def randomInUnitSphere(): Vector3 =
    val rnd = ThreadLocalRandom.current()
    while true do
      val x = rnd.nextDouble(-1, 1)
      val y = rnd.nextDouble(-1, 1)
      val z = rnd.nextDouble(-1, 1)
      if x * x + y * y + z * z < 1 then return Vector3(x, y, z)
    Vector3(0, 0, 0)

  def randomUnitVector(): Vector3 = randomInUnitSphere().normalized
