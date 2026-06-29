
How to run:

```shell
cd native
./compile.sh
```

```shell
sbt "Jmh/run"
```

```shell
sbt "Jmh/run Matrix4x4Benchmark"
```

in case of errors may help
```
sbt clean
```

## Results



AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S
Amazon.com Inc. Java 25.0.1

```
[info] Benchmark                                      Mode  Cnt    Score     Error  Units
[info] Matrix4x4Benchmark.downcallWithUpCall          avgt    5   24.003 ±   0.710  ns/op
[info] Matrix4x4Benchmark.getZero                     avgt    5    0.203 ±   0.004  ns/op
[info] Matrix4x4Benchmark.getZeroNative               avgt    5    6.562 ±   0.185  ns/op
[info] Matrix4x4Benchmark.multiply                    avgt    5  350.941 ±  17.191  ns/op
[info] Matrix4x4Benchmark.multiplyFastLoop            avgt    5   16.652 ±   0.376  ns/op
[info] Matrix4x4Benchmark.multiplyNative              avgt    5   24.836 ±   1.642  ns/op
[info] Matrix4x4Benchmark.multiplyNativeWithNewArena  avgt    5  278.400 ± 141.558  ns/op
```

AMD Ryzen 5950X
Amazon.com Inc. Java 25.0.1
```
[info] Benchmark                                      Mode  Cnt    Score    Error  Units
[info] Matrix4x4Benchmark.downcallWithUpCall          avgt   40   25.363 ±  0.071  ns/op
[info] Matrix4x4Benchmark.getZero                     avgt   40    0.439 ±  0.033  ns/op
[info] Matrix4x4Benchmark.getZeroNative               avgt   40    5.255 ±  0.083  ns/op
[info] Matrix4x4Benchmark.multiply                    avgt   40  496.855 ±  2.521  ns/op
[info] Matrix4x4Benchmark.multiplyFastLoop            avgt   40   17.892 ±  0.111  ns/op
[info] Matrix4x4Benchmark.multiplyFastRange           avgt   40   17.700 ±  0.043  ns/op
[info] Matrix4x4Benchmark.multiplyNative              avgt   40   27.260 ±  0.170  ns/op
[info] Matrix4x4Benchmark.multiplyNativeWithNewArena  avgt   40  529.496 ± 82.740  ns/op
[info] Matrix4x4Benchmark.multiplyOpaqueRange         avgt   40   17.623 ±  0.018  ns/op
```


In short:
* C function call overhead about 7 nanoseconds.
* Arena allocation is relatively slow and leads to something about 200 nanoseconds, better to reuse segments between calls.
* JIT cannot handle nested for loops in Scala


Surprisingly, scala compiler does some optimizations, and FastRange is not created at all:

```
// class version 61.0 (61)
// access flags 0x21
public class com/github/kright/MultiplyFastRange {

  // compiled from: Matrix4x4.scala

  ATTRIBUTE Scala : unknown

  ATTRIBUTE TASTY : unknown

  // access flags 0x1
  public <init>()V
   L0
    LINENUMBER 145 L0
    ALOAD 0
    INVOKESPECIAL java/lang/Object.<init> ()V
    RETURN
   L1
    LOCALVARIABLE this Lcom/github/kright/MultiplyFastRange; L0 L1 0
    MAXSTACK = 1
    MAXLOCALS = 1

  // access flags 0x1
  public multiplyFastRange(Lcom/github/kright/Matrix4x4;Lcom/github/kright/Matrix4x4;Lcom/github/kright/Matrix4x4;)V
    // parameter final  a
    // parameter final  b
    // parameter final  result
   L0
    LINENUMBER 149 L0
    ICONST_4
    ISTORE 4
   L1
    LINENUMBER 149 L1
    ICONST_0
    ISTORE 5
   L2
    ILOAD 5
    ILOAD 4
    IF_ICMPGE L3
   L4
    LINENUMBER 149 L4
    ILOAD 5
    ISTORE 6
   L5
    LINENUMBER 150 L5
    ICONST_4
    ISTORE 7
   L6
    LINENUMBER 150 L6
    ICONST_0
    ISTORE 8
   L7
    ILOAD 8
    ILOAD 7
    IF_ICMPGE L8
   L9
    LINENUMBER 150 L9
    ILOAD 8
    ISTORE 9
   L10
    LINENUMBER 151 L10
    DCONST_0
    DSTORE 10
   L11
    LINENUMBER 152 L11
    ICONST_4
    ISTORE 12
   L12
    LINENUMBER 152 L12
    ICONST_0
    ISTORE 13
   L13
    ILOAD 13
    ILOAD 12
    IF_ICMPGE L14
   L15
    LINENUMBER 152 L15
    ILOAD 13
    ISTORE 14
   L16
    LINENUMBER 153 L16
    DLOAD 10
    ALOAD 1
    ILOAD 6
    ILOAD 14
    INVOKEVIRTUAL com/github/kright/Matrix4x4.apply (II)D
    ALOAD 2
    ILOAD 14
    ILOAD 9
    INVOKEVIRTUAL com/github/kright/Matrix4x4.apply (II)D
    DMUL
    DADD
    DSTORE 10
   L17
    LINENUMBER 152 L17
    IINC 13 1
    GOTO L13
   L14
    LINENUMBER 155 L14
    ALOAD 3
    ILOAD 6
    ILOAD 9
    DLOAD 10
    INVOKEVIRTUAL com/github/kright/Matrix4x4.update (IID)V
   L18
    LINENUMBER 150 L18
    IINC 8 1
    GOTO L7
   L8
    LINENUMBER 149 L8
    IINC 5 1
    GOTO L2
   L3
    RETURN
   L19
    LOCALVARIABLE i I L13 L14 13
    LOCALVARIABLE FastRange_this I L12 L14 12
    LOCALVARIABLE sum D L11 L18 10
    LOCALVARIABLE i I L7 L8 8
    LOCALVARIABLE FastRange_this I L6 L8 7
    LOCALVARIABLE i I L2 L19 5
    LOCALVARIABLE FastRange_this I L1 L19 4
    LOCALVARIABLE this Lcom/github/kright/MultiplyFastRange; L0 L19 0
    LOCALVARIABLE a Lcom/github/kright/Matrix4x4; L0 L19 1
    LOCALVARIABLE b Lcom/github/kright/Matrix4x4; L0 L19 2
    LOCALVARIABLE result Lcom/github/kright/Matrix4x4; L0 L19 3
    MAXSTACK = 7
    MAXLOCALS = 15
}
```