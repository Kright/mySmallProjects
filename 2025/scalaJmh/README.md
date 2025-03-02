Jmh results:

```
Benchmark                                       Mode  Cnt  Score    Error  Units
BenchMatrixMultiply.multiplyCustomLoop          avgt    3  0.031 ? 0.002  us/op
BenchMatrixMultiply.multiplyFmaCustomLoop       avgt    3  0.026 ? 0.002  us/op
BenchMatrixMultiply.multiplyFmaFastRange        avgt    3  0.027 ? 0.004  us/op
BenchMatrixMultiply.multiplyFmaFullyUnrolled    avgt    3  0.024 ? 0.003  us/op
BenchMatrixMultiply.multiplyNaive               avgt    3  0.611 ? 0.027  us/op
BenchMatrixMultiply.multiplyNaiveFullyUnrolled  avgt    3  0.025 ? 0.002  us/op
BenchMatrixMultiply.multiplyNaiveUnrolled       avgt    3  0.039 ? 0.012  us/op
```