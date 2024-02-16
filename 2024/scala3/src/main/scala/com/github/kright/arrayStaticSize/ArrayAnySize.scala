package com.github.kright.arrayStaticSize

import scala.compiletime.ops.int.+
import scala.compiletime.{constValue, erasedValue}
import scala.reflect.ClassTag


class ArrayAnySize[T](val array: Array[T]):
  inline def ++[Sz2](other: ArrayAnySize[T])(using ClassTag[T]): ArrayAnySize[T] =
    new ArrayAnySize[T](array ++ other.array)


class ArrayStaticSize[T, Sz <: Int](array: Array[T]) extends ArrayAnySize[T](array):
  inline def staticSize: Int = constValue[Sz]

  inline def ++[Sz2 <: Int](other: ArrayStaticSize[T, Sz2])(using ClassTag[T]): ArrayStaticSize[T, +[Sz, Sz2]] =
    new ArrayStaticSize[T, +[Sz, Sz2]](array ++ other.array)

  inline def strictZip[T2](other: ArrayStaticSize[T2, Sz])(using ClassTag[T2], ClassTag[T]) =
    new ArrayStaticSize[(T, T2), Sz](array.zip(other.array))


object ArrayAnySize:
  inline def apply[T: ClassTag, Size <: Int](): ArrayStaticSize[T, Size] =
    inline constValue[Size] match
      case size: Int if size >= 0 => new ArrayStaticSize[T, Size](new Array[T](size))

  inline def apply[T](array: Array[T]): ArrayAnySize[T] =
    new ArrayAnySize[T](array)
