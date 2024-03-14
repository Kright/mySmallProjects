

JMH library with examples: [https://github.com/sbt/sbt-jmh](https://github.com/sbt/sbt-jmh)


Results (Maybe they are not actual, just for history):
FibonacciBench.measureNative                               sample  491094     1932.335 ±   1.738  ns/op
FibonacciBench.measureShadowStack                          sample  481714     3390.528 ±   3.061  ns/op
FibonacciBench.hardcodedMicroInterpreterFib                sample  478848     3929.257 ±   6.229  ns/op
FibonacciBench.realMicroInterpreterFibOptimized            sample  331421    11295.674 ±   7.040  ns/op
FibonacciBench.realMicroInterpreterFibOptimizedV2          sample  311646    12007.878 ±  12.118  ns/op
FibonacciBench.realMicroInterpreterFib                     sample  168409    44418.132 ± 122.825  ns/op

The difference between realMicroInterpreterFib and realMicroInterpreterFibOptimized that first is using array of interpreter instructions, but optimized version contains hard-coded class:

```Scala
final case class InstructionsSequence4(m0: MicroInterpreter,
                                       m1: MicroInterpreter,
                                       m2: MicroInterpreter,
                                       m3: MicroInterpreter) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    m0(stack)
    m1(stack)
    m2(stack)
    m3(stack)
```

I'm not sure that results are correct, but they are definitely interesting.
