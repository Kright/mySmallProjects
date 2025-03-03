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
BenchMatrixMultiply.multiplyCfor                avgt    3  0.030 ? 0.001  us/op
```

```
Benchmark                                       Mode  Cnt  Score   Error  Units
BenchForMotor.multiplyMotor                     avgt    3  0.014 ? 0.002  us/op
BenchForMotor.multiplyMotor3x                   avgt    3  0.041 ? 0.005  us/op
BenchForMotor.multiplyMotorRecord3x             avgt    3  0.041 ? 0.012  us/op
```