## IDEA project with JMH benchmark for value classes

Results on Ryzen 3700x
```
# JMH version: 1.37
# VM version: JDK 23-valhalla, OpenJDK 64-Bit Server VM, 23-valhalla+1-90

Benchmark                             Mode  Cnt      Score     Error  Units
BenchValhalla.finalPoint              avgt    5   3549.832 ± 563.635  us/op
BenchValhalla.plainOldMutablePoint    avgt    5  51994.276 ± 285.300  us/op
BenchValhalla.plainOldMutablePointV2  avgt    5  43164.733 ± 399.160  us/op
BenchValhalla.recordPoint             avgt    5   3358.368 ±  23.292  us/op
BenchValhalla.valhallaPoint           avgt    5   3723.338 ±  56.496  us/op
```