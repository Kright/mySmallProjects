package com.github.kright.vectinline

trait IVector2[T]:
  def x: T
  def y: T


trait IVector2Mut[T] extends IVector2[T]:
  var x: T
  var y: T


class Vector2d(var x: Double, var y: Double) extends IVector2Mut[Double]
class Vector2i(var x: Int, var y: Int) extends IVector2Mut[Int]


type VecMut2d[T] = T match
  case Double => Vector2d
  case Int => Vector2i


trait MyNumeric[T]:
  inline def add(a: T, b: T): T
  inline def mul(a: T, b: T): T


inline given a: MyNumeric[Double] with
  override inline def add(a: Double, b: Double): Double = a + b
  override inline def mul(a: Double, b: Double): Double = a * b


inline given ab: MyNumeric[Int] with
  override inline def add(a: Int, b: Int): Int = a + b
  override inline def mul(a: Int, b: Int): Int = a * b


extension[T, V <: IVector2Mut[T]] (v: V)
  inline def mag2(using inline num: MyNumeric[T]): T =
    val x = v.x
    val y = v.y
    num.add(num.mul(x, x), num.mul(y, y))

  inline def add[V2 <: IVector2[T]](r: V2)(using inline num: MyNumeric[T]): Unit =
    v.x = num.add(v.x, r.x)
    v.y = num.add(v.y, r.y)


@main
def main(): Unit =
  Main().main(Vector2d(1, 2), Vector2d(3, 4))

class Main:
  def main(v: VecMut2d[Double], v2: VecMut2d[Double]) =
    v.add(v2)

  def xz() =
    val a = new VecMut2d[Double](0, 0)
