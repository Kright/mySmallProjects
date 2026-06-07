package me.kright.raytracer

import java.util.concurrent.ThreadLocalRandom
import scala.IArray

object Scene:
  def generateRandomScene(): HittableList = {
    val spheres = scala.collection.mutable.ArrayBuffer.empty[Hittable]

    // Ground
    spheres += Sphere(Vector3(0, -1000, 0), 1000, Lambertian(Color(0.5, 0.5, 0.5)))

    val rnd = ThreadLocalRandom.current()

    for (a <- -11 until 11; b <- -11 until 11) {
      val chooseMat = rnd.nextDouble()
      val center = Vector3(a + 0.9 * rnd.nextDouble(), 0.2, b + 0.9 * rnd.nextDouble())

      if ((center - Vector3(4, 0.2, 0)).length > 0.9) {
        if (chooseMat < 0.8) {
          // diffuse
          val albedo = Color(rnd.nextDouble() * rnd.nextDouble(), rnd.nextDouble() * rnd.nextDouble(), rnd.nextDouble() * rnd.nextDouble())
          spheres += Sphere(center, 0.2, Lambertian(albedo))
        } else if (chooseMat < 0.95) {
          // metal
          val albedo = Color(rnd.nextDouble(0.5, 1), rnd.nextDouble(0.5, 1), rnd.nextDouble(0.5, 1))
          val fuzz = rnd.nextDouble(0, 0.5)
          spheres += Sphere(center, 0.2, Metal(albedo, fuzz))
        } else {
          // glass
          spheres += Sphere(center, 0.2, Dielectric(1.5))
        }
      }
    }

    spheres += Sphere(Vector3(0, 1, 0), 1.0, Dielectric(1.5))
    spheres += Sphere(Vector3(-4, 1, 0), 1.0, Lambertian(Color(0.4, 0.2, 0.1)))
    spheres += Sphere(Vector3(4, 1, 0), 1.0, Metal(Color(0.7, 0.6, 0.5), 0.0))

    HittableList(IArray.from(spheres))
  }
