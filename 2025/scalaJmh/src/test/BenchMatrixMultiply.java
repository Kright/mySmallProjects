package test;

import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;


@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(value = 1)
@Warmup(iterations = 3, timeUnit = TimeUnit.MILLISECONDS, time = 5000)
@Measurement(iterations = 3, timeUnit = TimeUnit.MILLISECONDS, time = 5000)
public class BenchMatrixMultiply {
    private Matrix4x4 matrixA, matrixB;

    @Setup
    public void setup() {
        matrixA = new Matrix4x4();
        matrixB = new Matrix4x4();
        matrixA.fillRandom();
        matrixB.fillRandom();
    }

    @Benchmark
    public Matrix4x4 multiplyNaive() {
        return matrixA.multiplyNaive(matrixB);
    }

    @Benchmark
    public Matrix4x4 multiplyNaiveUnrolled() {
        return matrixA.multiplyNaiveUnrolled(matrixB);
    }

    @Benchmark
    public Matrix4x4 multiplyNaiveFullyUnrolled() {
        return matrixA.multiplyNaiveFullyUnrolled(matrixB);
    }

    @Benchmark
    public Matrix4x4 multiplyFmaFullyUnrolled() {
        return matrixA.multiplyFmaFullyUnrolled(matrixB);
    }

    @Benchmark
    public Matrix4x4 multiplyCustomLoop() {
        return matrixA.multiplyCustomLoop(matrixB);
    }

    @Benchmark
    public Matrix4x4 multiplyFmaCustomLoop() {
        return matrixA.multiplyFmaCustomLoop(matrixB);
    }

   @Benchmark
    public Matrix4x4 multiplyFmaFastRange() {
        return matrixA.multiplyFmaFastRange(matrixB);
    }

    @Benchmark
    public Matrix4x4 multiplyCfor() {
        return matrixA.multiplyCfor(matrixB);
    }
}
