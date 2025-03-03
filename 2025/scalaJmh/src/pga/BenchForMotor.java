package pga;

import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(value = 1)
@Warmup(iterations = 3, timeUnit = TimeUnit.MILLISECONDS, time = 5000)
@Measurement(iterations = 3, timeUnit = TimeUnit.MILLISECONDS, time = 5000)
public class BenchForMotor {
    private Pga3dMotor motorA, motorB, motorC, motorD;
    private Pga3dMotorRecord motorAr, motorBr, motorCr, motorDr;

    private Pga3dMotor makeRandomMotor() {
        return new Pga3dMotor(Math.random(), Math.random(), Math.random(), Math.random(), Math.random(), Math.random(), Math.random(), Math.random());
    }

    private Pga3dMotorRecord makeRandomMotorRecord() {
        return new Pga3dMotorRecord(Math.random(), Math.random(), Math.random(), Math.random(), Math.random(), Math.random(), Math.random(), Math.random());
    }
    
    @Setup
    public void setup() {
        motorA = makeRandomMotor();
        motorB = makeRandomMotor();
        motorC = makeRandomMotor();
        motorD = makeRandomMotor();
        motorAr = makeRandomMotorRecord();
        motorBr = makeRandomMotorRecord();
        motorCr = makeRandomMotorRecord();
        motorDr = makeRandomMotorRecord();
    }

    @Benchmark
    public Pga3dMotor multiplyMotor3x() {
        return motorA.geometric(motorB).geometric(motorC).geometric(motorD);
    }

    @Benchmark
    public Pga3dMotor multiplyMotor() {
        return motorA.geometric(motorB);
    }

    @Benchmark
    public Pga3dMotorRecord multiplyMotorRecord3x() {
        return motorAr.geometric(motorBr).geometric(motorCr).geometric(motorDr);
    }
}
