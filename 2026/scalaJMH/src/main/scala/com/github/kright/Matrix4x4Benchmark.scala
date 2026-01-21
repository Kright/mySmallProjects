package com.github.kright

import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.util.concurrent.TimeUnit

@State(Scope.Thread)
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
class Matrix4x4Benchmark {

  var matrixA: Matrix4x4 = _
  var matrixB: Matrix4x4 = _
  var result: Matrix4x4 = _

  var nativeMultiplier: NativeMultiplier = _

  @Setup
  def setup(): Unit = {
    matrixA = Matrix4x4.random()
    matrixB = Matrix4x4.random()
    result = Matrix4x4()

    nativeMultiplier = new NativeMultiplier()
  }

  @Benchmark
  def multiply(bh: Blackhole): Unit = {
    com.github.kright.multiply(matrixA, matrixB, result)
    bh.consume(result)
  }

  @Benchmark
  def multiplyFastLoop(bh: Blackhole): Unit = {
    com.github.kright.multiplyFastLoop(matrixA, matrixB, result)
    bh.consume(result)
  }

  @Benchmark
  def multiplyNative(bh: Blackhole): Unit = {
    nativeMultiplier.multiply(matrixA, matrixB, result)
    bh.consume(result)
  }

  @Benchmark
  def multiplyNativeWithNewArena(bh: Blackhole): Unit = {
    nativeMultiplier.multiplyWithNewArea(matrixA, matrixB, result)
    bh.consume(result)
  }

  @Benchmark
  def getZeroNative(bh: Blackhole): Unit = {
    val value: Double = nativeMultiplier.getZeroHandle.invoke()
    bh.consume(value)
  }

  @Benchmark
  def getZero(bh: Blackhole): Unit = {
    val value: Double = getZeroDouble()
    bh.consume(value)
  }
  
  @Benchmark()
  def downcallWithUpCall(bh: Blackhole): Unit = {
    bh.consume(nativeMultiplier.downcallWithUpCall())
  }
}

private def getZeroDouble(): Double = 0.0
