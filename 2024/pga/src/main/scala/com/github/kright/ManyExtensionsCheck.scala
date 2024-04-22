package com.github.kright

import scala.annotation.targetName
import scala.math.Numeric.Implicits.infixNumericOps

@main
def xz(): Unit = {
}

class ManyExtensionsCheck:
  def run2d(): Double = {
    import ManyExtensionsCheck.*
    val a = Vec2d(1.0, 2.0)
    val b = Vec2d(10.0, 20.0)
    a.dot(b)
  }


object ManyExtensionsCheck:
  trait IVec3[T]:
    def x: T
    def y: T
    def z: T

  case class Vec3d(x: Double, y: Double, z: Double) extends IVec3[Double]

  case class Vec2d(x: Double, y: Double) extends IVec3[Double]:
    override inline def z: Double = 0.0

  extension (a: Vec2d) @targetName("dotVec2dVec2d") def dot(b: Vec2d): Double = a.x * b.x + a.y * b.y
  extension (a: Vec2d) @targetName("dotVec2dVec3d") def dot(b: Vec3d): Double = a.x * b.x + a.y * b.y
  extension (a: Vec3d) @targetName("dotVec3dVec2d") def dot(b: Vec2d): Double = a.x * b.x + a.y * b.y
  extension (a: Vec3d) @targetName("dotVec3dVec3d") def dot(b: Vec3d): Double = a.x * b.x + a.y * b.y + a.z * b.z

  extension[T: Numeric] (a: IVec3[T]) @targetName("dotIVec3IVec3") def dot(b: IVec3[T]): T = a.x * b.x + a.y * b.y + a.z * b.z