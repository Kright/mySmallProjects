package test;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;

public class MyBench {
    static final int samples = 5000;
    static final ValuePoint[] valhallaPoint = new ValuePoint[samples];
    static final MutablePoint[] mutablePoint = new MutablePoint[samples];
    static final RecordPoint[] recordPoint = new RecordPoint[samples];
    static final FinalPoint[] finalPoint = new FinalPoint[samples];

    static {
        var r = new Random(69);
        for (int i = 0; i < valhallaPoint.length; i++) {
            valhallaPoint[i] = new ValuePoint(r.nextDouble(), r.nextDouble(), r.nextDouble(), r.nextDouble());
        }
        for (int i = 0; i < mutablePoint.length; i++) {
            var valh = valhallaPoint[i];
            mutablePoint[i] = new MutablePoint(valh.x, valh.y, valh.z, valh.w);
        }
    }

    @Benchmark
    public void measureName(Blackhole bh) {
        var arr = valhallaPoint;
        var sum = new ValuePoint(0, 0, 0, 0);
        for (int i = 0; i < arr.length; ++i) {
            for (int j = 0; j < arr.length; ++j) {
                sum.add(arr[j].sub(arr[i]));
            }
        }
        bh.consume(sum);
    }
}
