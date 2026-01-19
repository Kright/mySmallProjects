
How to run:

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
[info] Matrix4x4Benchmark.getZero                     avgt    5    0.208 ±   0.043  ns/op
[info] Matrix4x4Benchmark.getZeroNative               avgt    5    6.585 ±   0.367  ns/op
[info] Matrix4x4Benchmark.multiply                    avgt    5  359.286 ±  22.784  ns/op
[info] Matrix4x4Benchmark.multiplyFastLoop            avgt    5   16.727 ±   0.513  ns/op
[info] Matrix4x4Benchmark.multiplyNative              avgt    5   20.693 ±   1.053  ns/op
[info] Matrix4x4Benchmark.multiplyNativeWithNewArena  avgt    5  267.742 ± 138.956  ns/op
```

In short:
* C function call overhead about 7 nanoseconds.
* Arena allocation is relatively slow and leads to something about 200 nanoseconds, better to reuse segments between calls.
* JIT cannot handle nested for loops in Scala