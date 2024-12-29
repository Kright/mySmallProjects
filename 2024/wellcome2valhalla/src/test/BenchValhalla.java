package test;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;


@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(value = 1)
@Warmup(iterations = 5, timeUnit = TimeUnit.MILLISECONDS, time = 5000)
@Measurement(iterations = 5, timeUnit = TimeUnit.MILLISECONDS, time = 5000)
public class BenchValhalla {
    private MutablePoint[] mutablePoint;
    private ValuePoint[] valhallaPoint;
    private RecordPoint[] recordPoint;
    private FinalPoint[] finalPoint;

    @Setup
    public void setup() {
        var r = new Random(69);
        var samples = 5000;
        this.valhallaPoint = new ValuePoint[samples];
        this.mutablePoint = new MutablePoint[samples];
        this.recordPoint = new RecordPoint[samples];
        this.finalPoint = new FinalPoint[samples];

        for (int i = 0; i < valhallaPoint.length; i++) {
            valhallaPoint[i] = new ValuePoint(r.nextDouble(), r.nextDouble(), r.nextDouble(), r.nextDouble());
        }
        for (int i = 0; i < mutablePoint.length; i++) {
            var valh = valhallaPoint[i];
            mutablePoint[i] = new MutablePoint(valh.x, valh.y, valh.z, valh.w);
            recordPoint[i] = new RecordPoint(valh.x, valh.y, valh.z, valh.w);
            finalPoint[i] = new FinalPoint(valh.x, valh.y, valh.z, valh.w);
        }
    }

    @Benchmark
    public ValuePoint valhallaPoint() {
        var arr = valhallaPoint;
        var sum = new ValuePoint(0, 0, 0, 0);
        for (int i = 0; i < arr.length; ++i) {
            for (int j = 0; j < arr.length; ++j) {
                sum.add(arr[j].sub(arr[i]));
            }
        }
        return sum;
    }

    @Benchmark
    public FinalPoint finalPoint() {
        var arr = finalPoint;
        var sum = new FinalPoint(0, 0, 0, 0);
        for (int i = 0; i < arr.length; ++i) {
            for (int j = 0; j < arr.length; ++j) {
                sum.add(arr[j].sub(arr[i]));
            }
        }
        return sum;
    }

    @Benchmark
    public RecordPoint recordPoint() {
        var arr = recordPoint;
        var sum = new RecordPoint(0, 0, 0, 0);
        for (int i = 0; i < arr.length; ++i) {
            for (int j = 0; j < arr.length; ++j) {
                sum.add(arr[j].sub(arr[i]));
            }
        }
        return sum;
    }

    @Benchmark
    public MutablePoint plainOldMutablePoint() {
        var arr = mutablePoint;
        var sum = new MutablePoint(0, 0, 0, 0);
        for (int i = 0; i < arr.length; ++i) {
            for (int j = 0; j < arr.length; ++j) {
                sum.add(arr[j]);
                sum.sub(arr[i]);
            }
        }
        return sum;
    }

    @Benchmark
    public MutablePoint plainOldMutablePointV2() {
        var arr = mutablePoint;
        var sum = new MutablePoint(0, 0, 0, 0);
        for (int i = 0; i < arr.length; ++i) {
            for (int j = 0; j < arr.length; ++j) {
                sum.add(arr[j].copy().sub(arr[i]));
            }
        }
        return sum;
    }

    public static void main(String[] args) throws Exception {
        var options = new OptionsBuilder()
                .include(BenchValhalla.class.getSimpleName())
                .build();
        new Runner(options).run();
    }
}
