package com.github.kright

import java.lang.foreign.MemorySegment.ofArray
import java.lang.foreign.{Arena, FunctionDescriptor, Linker, SymbolLookup, ValueLayout}
import java.lang.invoke.MethodHandle


class NativeMultiplier {
  val arena: Arena = Arena.ofConfined()
  val aSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, 16L)
  val bSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, 16L)
  val resultSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, 16L)

  System.load(java.io.File("native/libcode.so").getAbsolutePath)

  val linker = Linker.nativeLinker()
  val lookup = SymbolLookup.loaderLookup()
  val symbolOpt = lookup.find("matrix4x4_multiply").get()

  val matrixMultiplyHandle: MethodHandle = linker.downcallHandle(
    symbolOpt,
    FunctionDescriptor.ofVoid(
      ValueLayout.ADDRESS,
      ValueLayout.ADDRESS,
      ValueLayout.ADDRESS
    )
  )

  def multiply(a: Matrix4x4, b: Matrix4x4, result: Matrix4x4): Unit = {
    aSegment.copyFrom(ofArray(a.data))
    bSegment.copyFrom(ofArray(b.data))
    matrixMultiplyHandle.invoke(aSegment, bSegment, resultSegment)
    ofArray(result.data).copyFrom(resultSegment)
  }

  def multiplyWithNewArea(a: Matrix4x4, b: Matrix4x4, result: Matrix4x4): Unit = {
    val newArena = Arena.ofConfined()

    try {
      val aSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, 16L)
      val bSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, 16L)
      val resultSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, 16L)
      aSegment.copyFrom(ofArray(a.data))
      bSegment.copyFrom(ofArray(b.data))
      // Call native function
      matrixMultiplyHandle.invoke(aSegment, bSegment, resultSegment)
      ofArray(result.data).copyFrom(resultSegment)
    }
    finally {
      newArena.close()
    }
  }
}
