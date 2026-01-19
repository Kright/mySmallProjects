
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

[info] Benchmark                                      Mode  Cnt    Score     Error  Units
[info] Matrix4x4Benchmark.multiply                    avgt    5  361.011 ±  27.590  ns/op
[info] Matrix4x4Benchmark.multiplyFastLoop            avgt    5   16.955 ±   0.384  ns/op
[info] Matrix4x4Benchmark.multiplyNative              avgt    5   21.144 ±   1.145  ns/op
[info] Matrix4x4Benchmark.multiplyNativeWithNewArena  avgt    5  283.425 ± 139.326  ns/op
