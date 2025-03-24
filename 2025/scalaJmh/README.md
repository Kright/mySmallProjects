# Jmh results:

## Ryzen 3700

openjdk 23

```
Benchmark                                       Mode  Cnt  Score    Error  Units
BenchMatrixMultiply.multiplyCforFma             avgt    3  0.023 ?  0.001  us/op
BenchMatrixMultiply.multiplyFmaCustomLoop       avgt    3  0.023 ?  0.003  us/op
BenchMatrixMultiply.multiplyNaiveFullyUnrolled  avgt    3  0.024 ?  0.004  us/op
BenchMatrixMultiply.multiplyFmaFastRange        avgt    3  0.024 ?  0.001  us/op
BenchMatrixMultiply.multiplyFmaFullyUnrolled    avgt    3  0.024 ?  0.001  us/op
BenchMatrixMultiply.multiplyCfor                avgt    3  0.027 ?  0.001  us/op
BenchMatrixMultiply.multiplyCustomLoop          avgt    3  0.027 ?  0.001  us/op
BenchMatrixMultiply.multiplyNaiveUnrolled       avgt    3  0.045 ?  0.004  us/op
BenchMatrixMultiply.multiplyNaive               avgt    3  0.615 ?  0.023  us/op
```

```
Benchmark                                       Mode  Cnt  Score    Error  Units
BenchForMotor.multiplyMotor                     avgt    3  0.014 ?  0.001  us/op
BenchForMotor.multiplyMotorWithFma              avgt    3  0.014 ?  0.001  us/op
BenchForMotor.multiplyMotor3x                   avgt    3  0.040 ?  0.001  us/op
BenchForMotor.multiplyMotorRecord3x             avgt    3  0.040 ?  0.005  us/op
BenchForMotor.multiplyMotorWithFma3x            avgt    3  0.044 ?  0.007  us/op
```

graalvm ce 23.0.2

```
BenchMatrixMultiply.multiplyCforFma             avgt    3  0.016 ?  0.002  us/op
BenchMatrixMultiply.multiplyFmaCustomLoop       avgt    3  0.017 ?  0.003  us/op
BenchMatrixMultiply.multiplyFmaFullyUnrolled    avgt    3  0.017 ?  0.001  us/op
BenchMatrixMultiply.multiplyNaiveFullyUnrolled  avgt    3  0.020 ?  0.001  us/op
BenchMatrixMultiply.multiplyCfor                avgt    3  0.021 ?  0.001  us/op
BenchMatrixMultiply.multiplyCustomLoop          avgt    3  0.021 ?  0.002  us/op
BenchMatrixMultiply.multiplyFmaFastRange        avgt    3  0.024 ?  0.001  us/op
BenchMatrixMultiply.multiplyNaiveUnrolled       avgt    3  0.050 ?  0.011  us/op
BenchMatrixMultiply.multiplyNaive               avgt    3  0.061 ?  0.033  us/op
```

```
Benchmark                                       Mode  Cnt  Score   Error  Units
BenchForMotor.multiplyMotor                     avgt    3  0.010 ? 0.001  us/op
BenchForMotor.multiplyMotorWithFma              avgt    3  0.011 ? 0.001  us/op
BenchForMotor.multiplyMotor3x                   avgt    3  0.028 ? 0.001  us/op
BenchForMotor.multiplyMotorRecord3x             avgt    3  0.028 ? 0.002  us/op
BenchForMotor.multiplyMotorWithFma3x            avgt    3  0.037 ? 0.004  us/op
```

openjdk 21.0.6

```
Benchmark                                       Mode  Cnt  Score   Error  Units
BenchMatrixMultiply.multiplyCforFma             avgt    3  0.023 ± 0.001  us/op
BenchMatrixMultiply.multiplyFmaCustomLoop       avgt    3  0.023 ± 0.002  us/op
BenchMatrixMultiply.multiplyFmaFastRange        avgt    3  0.024 ± 0.001  us/op
BenchMatrixMultiply.multiplyFmaFullyUnrolled    avgt    3  0.024 ± 0.004  us/op
BenchMatrixMultiply.multiplyNaiveFullyUnrolled  avgt    3  0.025 ± 0.002  us/op
BenchMatrixMultiply.multiplyCustomLoop          avgt    3  0.026 ± 0.001  us/op
BenchMatrixMultiply.multiplyCfor                avgt    3  0.027 ± 0.001  us/op
BenchMatrixMultiply.multiplyNaiveUnrolled       avgt    3  0.038 ± 0.004  us/op
BenchMatrixMultiply.multiplyNaive               avgt    3  0.602 ± 0.098  us/op

```

## Ryzen 5950

openjdk 21.0.6

```
Benchmark                                       Mode  Cnt  Score    Error  Units
BenchMatrixMultiply.multiplyCforFma             avgt    3  0.018 ±  0.005  us/op
BenchMatrixMultiply.multiplyFmaCustomLoop       avgt    3  0.019 ±  0.001  us/op
BenchMatrixMultiply.multiplyCfor                avgt    3  0.020 ±  0.001  us/op
BenchMatrixMultiply.multiplyFmaFastRange        avgt    3  0.020 ±  0.004  us/op
BenchMatrixMultiply.multiplyFmaFullyUnrolled    avgt    3  0.020 ±  0.001  us/op
BenchMatrixMultiply.multiplyNaiveFullyUnrolled  avgt    3  0.020 ±  0.002  us/op
BenchMatrixMultiply.multiplyCustomLoop          avgt    3  0.021 ±  0.001  us/op
BenchMatrixMultiply.multiplyNaiveUnrolled       avgt    3  0.028 ±  0.001  us/op
BenchMatrixMultiply.multiplyNaive               avgt    3  0.449 ±  0.036  us/op
```

openjdk 23.0.1

```
Benchmark                                       Mode  Cnt  Score    Error  Units
BenchMatrixMultiply.multiplyFmaCustomLoop       avgt    3  0.018 ?  0.003  us/op
BenchMatrixMultiply.multiplyCforFma             avgt    3  0.019 ?  0.001  us/op
BenchMatrixMultiply.multiplyFmaFullyUnrolled    avgt    3  0.019 ?  0.003  us/op
BenchMatrixMultiply.multiplyNaiveFullyUnrolled  avgt    3  0.019 ?  0.001  us/op
BenchMatrixMultiply.multiplyFmaFastRange        avgt    3  0.020 ?  0.001  us/op
BenchMatrixMultiply.multiplyCustomLoop          avgt    3  0.021 ?  0.007  us/op
BenchMatrixMultiply.multiplyCfor                avgt    3  0.021 ?  0.001  us/op
BenchMatrixMultiply.multiplyNaiveUnrolled       avgt    3  0.030 ?  0.005  us/op
BenchMatrixMultiply.multiplyNaive               avgt    3  0.459 ?  0.015  us/op
```

openjdk 24

```
Benchmark                                       Mode  Cnt  Score    Error  Units
BenchMatrixMultiply.multiplyCforFma             avgt    3  0.018 ±  0.001  us/op
BenchMatrixMultiply.multiplyFmaCustomLoop       avgt    3  0.018 ±  0.001  us/op
BenchMatrixMultiply.multiplyFmaFastRange        avgt    3  0.019 ±  0.001  us/op
BenchMatrixMultiply.multiplyFmaFullyUnrolled    avgt    3  0.019 ±  0.001  us/op
BenchMatrixMultiply.multiplyNaiveFullyUnrolled  avgt    3  0.019 ±  0.001  us/op
BenchMatrixMultiply.multiplyCfor                avgt    3  0.020 ±  0.001  us/op
BenchMatrixMultiply.multiplyCustomLoop          avgt    3  0.020 ±  0.002  us/op
BenchMatrixMultiply.multiplyNaiveUnrolled       avgt    3  0.028 ±  0.003  us/op
BenchMatrixMultiply.multiplyNaive               avgt    3  0.458 ±  0.011  us/op
```