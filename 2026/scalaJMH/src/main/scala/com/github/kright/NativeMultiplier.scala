package com.github.kright

import java.lang.foreign.MemorySegment
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

  val getZeroHandle: MethodHandle = linker.downcallHandle(
    lookup.find("getDoubleZero").get(),
    FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE)
  )

  val downcallWithUpCallHandle: MethodHandle = linker.downcallHandle(
    lookup.find("callFunction").get(),
    FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS),
  )

  private def jvmGetZero(): Double = 0.0

  val jvmGetZeroStub: MemorySegment = linker.upcallStub(
    java.lang.invoke.MethodHandles.lookup().findVirtual(
      classOf[NativeMultiplier],
      "jvmGetZero",
      java.lang.invoke.MethodType.methodType(classOf[Double])
    ).bindTo(this),
    FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE),
    arena
  )

  def downcallWithUpCall(): Double = {
    downcallWithUpCallHandle.invoke(jvmGetZeroStub).asInstanceOf[Double]
  }

  def multiply(a: Matrix4x4, b: Matrix4x4, result: Matrix4x4): Unit = {
    aSegment.copyFrom(MemorySegment.ofArray(a.data))
    bSegment.copyFrom(MemorySegment.ofArray(b.data))
    matrixMultiplyHandle.invoke(aSegment, bSegment, resultSegment)
    MemorySegment.ofArray(result.data).copyFrom(resultSegment)
  }

  def multiplyWithNewArea(a: Matrix4x4, b: Matrix4x4, result: Matrix4x4): Unit = {
    val newArena = Arena.ofConfined()

    try {
      val aSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, 16L)
      val bSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, 16L)
      val resultSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, 16L)
      aSegment.copyFrom(MemorySegment.ofArray(a.data))
      bSegment.copyFrom(MemorySegment.ofArray(b.data))
      // Call native function
      matrixMultiplyHandle.invoke(aSegment, bSegment, resultSegment)
      MemorySegment.ofArray(result.data).copyFrom(resultSegment)
    }
    finally {
      newArena.close()
    }
  }

  def callGetZero(): Double = {
    getZeroHandle.invoke()
  }
}
