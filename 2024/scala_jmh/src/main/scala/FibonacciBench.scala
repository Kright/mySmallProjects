package com.github.kright

import org.openjdk.jmh.annotations.*

import java.util.concurrent.TimeUnit

object FibonacciBench:
  @State(Scope.Benchmark)
  class FibTask {
    val n: Int = 14

    val expected: Int = nativeFib(n)
  }

  def nativeFib(n: Int): Int =
    if (n <= 1) 1
    else nativeFib(n - 1) + nativeFib(n - 2)

@BenchmarkMode(Array(Mode.SampleTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1000, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 1000, timeUnit = TimeUnit.MILLISECONDS)
@Fork(3)
class FibonacciBench:

  import FibonacciBench.*

  private def shadowStackFib(stack: IntStack): Unit =
    val n = stack.head
    if (n <= 1) {
      stack.head = 1
    } else {
      stack.push(n - 1)
      shadowStackFib(stack)
      stack.push(n - 2)
      shadowStackFib(stack)
      val result = stack.pop() + stack.pop()
      stack.head = result
    }

  private def hardcodedMicroInterpreterFib(stack: IntStack): Unit =
    stack.head -= 1
    if (stack.head <= 0) {
      stack.head = 1
    } else {
      stack.dup(0)
      hardcodedMicroInterpreterFib(stack)
      stack.dup(1)
      stack.head -= 1
      hardcodedMicroInterpreterFib(stack)
      stack.add()
      stack.copyHead(1)
      stack.pop()
    }

  private def makeMicroInterpreterForFib(): MicroInterpreter =
    var m: MicroInterpreter = null

    m = InstructionsSequence(Array(
      DecHead,
      IfHeadLE(
        SetHead(1),
        InstructionsSequence(Array(
          Dup(0),
          Call(() => m),
          Dup(1),
          DecHead,
          Call(() => m),
          Add,
          CopyHead(1),
          Pop,
        ))
      )
    ))

    m

  private def makeMicroInterpreterForFibOptimized(): MicroInterpreter =
    var m: MicroInterpreter = null

    m = InstructionsSequence2(
      DecHead,
      IfHeadLE(
        SetHead(1),
        InstructionsSequence8(
          Dup(0),
          Call(() => m),
          Dup(1),
          DecHead,
          Call(() => m),
          Add,
          CopyHead(1),
          Pop,
        )
      )
    )

    m

  private def makeMicroInterpreterForFibOptimizedV2(): MicroInterpreter =
    var m: MicroInterpreter = null

    m = OptimizedInstructionSequence(
      DecHead,
      IfHeadLE(
        SetHead(1),
        OptimizedInstructionSequence(
          Dup(0),
          Call(() => m),
          Dup(1),
          DecHead,
          Call(() => m),
          Add,
          CopyHead(1),
          Pop,
        )
      )
    )

    m  

//  @Benchmark
  def realMicroInterpreterFib(task: FibTask): Int = {
    val stack = new IntStack(64)
    stack.push(task.n)
    val interpreter = makeMicroInterpreterForFib()
    interpreter(stack)
    require(stack.sPointer == 0)
    require(stack.head == task.expected)
    stack.head
  }

//  @Benchmark
  def realMicroInterpreterFibOptimized(task: FibTask): Int = {
    val stack = new IntStack(64)
    stack.push(task.n)
    val interpreter = makeMicroInterpreterForFibOptimized()
    interpreter(stack)
    require(stack.sPointer == 0)
    require(stack.head == task.expected)
    stack.head
  }
  
//  @Benchmark
  def realMicroInterpreterFibOptimizedV2(task: FibTask): Int = {
    val stack = new IntStack(64)
    stack.push(task.n)
    val interpreter = makeMicroInterpreterForFibOptimizedV2()
    interpreter(stack)
    require(stack.sPointer == 0)
    require(stack.head == task.expected)
    stack.head
  }

//  @Benchmark
  def measureNative(task: FibTask): Int = {
    nativeFib(task.n)
  }

//  @Benchmark
  def measureShadowStack(task: FibTask): Int = {
    val stack = new IntStack(64)
    stack.push(task.n)
    shadowStackFib(stack)
    require(stack.sPointer == 0)
    require(stack.head == task.expected)
    stack.head
  }

//  @Benchmark
  def hardcodedMicroInterpreterFib(task: FibTask): Int = {
    val stack = new IntStack(64)
    stack.push(task.n)
    hardcodedMicroInterpreterFib(stack)
    require(stack.sPointer == 0)
    require(stack.head == task.expected)
    stack.head
  }

  