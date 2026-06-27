
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
[info] Benchmark                                      Mode  Cnt    Score     Error  Units
[info] Matrix4x4Benchmark.downcallWithUpCall          avgt    5   25.552 ±   0.648  ns/op
[info] Matrix4x4Benchmark.getZero                     avgt    5    0.438 ±   0.038  ns/op
[info] Matrix4x4Benchmark.getZeroNative               avgt    5    5.974 ±   0.592  ns/op
[info] Matrix4x4Benchmark.multiply                    avgt    5  511.062 ±  99.651  ns/op
[info] Matrix4x4Benchmark.multiplyFastLoop            avgt    5   18.807 ±   2.142  ns/op
[info] Matrix4x4Benchmark.multiplyFastRange           avgt    5   17.991 ±   0.730  ns/op
[info] Matrix4x4Benchmark.multiplyNative              avgt    5   28.069 ±   3.660  ns/op
[info] Matrix4x4Benchmark.multiplyNativeWithNewArena  avgt    5  565.757 ± 739.723  ns/op
```


In short:
* C function call overhead about 7 nanoseconds.
* Arena allocation is relatively slow and leads to something about 200 nanoseconds, better to reuse segments between calls.
* JIT cannot handle nested for loops in Scala