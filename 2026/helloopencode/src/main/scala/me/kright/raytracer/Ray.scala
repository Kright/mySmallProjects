package me.kright.raytracer

case class Ray(origin: Vector3, direction: Vector3):
  def at(t: Double): Vector3 = origin + direction * t
