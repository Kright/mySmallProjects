package com.github.kright

import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit

import java.util.Random


@BenchmarkMode(Array(Mode.SampleTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1000, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 1000, timeUnit = TimeUnit.MILLISECONDS)
@Fork(3)
class RecordClassBench {
//  @Benchmark
  def sumOfVector3drec(data: RecordClassBench.VectorsArrays, blackhole: Blackhole): Unit = {
    var sum = Vector3dRec(0.0, 0.0, 0.0)
    for (v <- data.vecsRecs) {
      sum += v
    }
    blackhole.consume(sum)
  }

//  @Benchmark
  def sumOfVector3d(data: RecordClassBench.VectorsArrays, blackhole: Blackhole): Unit = {
    var sum = Vector3d(0.0, 0.0, 0.0)
    for (v <- data.vecs) {
      sum += v
    }
    blackhole.consume(sum)
  }

//  @Benchmark
  def sumOfVector3dMut(data: RecordClassBench.VectorsArrays, blackhole: Blackhole): Unit = {
    val sum = Vector3dMut(0.0, 0.0, 0.0)
    for (v <- data.vecsMut) {
      sum += v
    }
    blackhole.consume(sum)
  }

//  @Benchmark
  def sumOfVector3dMutManualLoop(data: RecordClassBench.VectorsArrays, blackhole: Blackhole): Unit = {
    val sum = Vector3dMut(0.0, 0.0, 0.0)
    val vecsMut = data.vecsMut
    forAll(vecsMut.length) { i =>
      sum += vecsMut(i)
    }
    blackhole.consume(sum)
  }
  
  private inline def forAll(count: Int)(inline func: Int => Unit): Unit =
    var i = 0
    while (i < count) {
      func(i)
      i += 1
    }
}


object RecordClassBench:
  @State(Scope.Benchmark)
  class VectorsArrays {
    val arrSize = 128

    val vecsRecs = new Array[Vector3dRec](arrSize)
    val vecs = new Array[Vector3d](arrSize)
    val vecsMut = new Array[Vector3dMut](arrSize)

    {
      val random = new Random()
      for (i <- 0 until arrSize) {
        val x = random.nextDouble()
        val y = random.nextDouble()
        val z = random.nextDouble()
        vecsRecs(i) = new Vector3dRec(x, y, z)
        vecs(i) = new Vector3d(x, y, z)
        vecsMut(i) = new Vector3dMut(x, y, z)
      }
    }
  }

private class Vector3dRec(val x: Double, val y: Double, val z: Double) extends Record:
  def +(other: Vector3dRec): Vector3dRec =
    new Vector3dRec(x + other.x, y + other.y, z + other.z)

  override def equals(obj: Any): Boolean = false
  override def hashCode(): Int = 0
  override def toString: String = ""

private final class Vector3d(val x: Double, val y: Double, val z: Double):
  def +(other: Vector3d): Vector3d =
    new Vector3d(x + other.x, y + other.y, z + other.z)

private class Vector3dMut(var x: Double, var y: Double, var z: Double):
  def +(other: Vector3dMut): Vector3dMut =
    new Vector3dMut(x + other.x, y + other.y, z + other.z)

  def +=(other: Vector3dMut): Unit =
    x += other.x
    y += other.y
    z += other.z