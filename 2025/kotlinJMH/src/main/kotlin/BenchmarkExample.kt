package org.example

import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.OutputTimeUnit
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.Level
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit
import kotlin.random.Random

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
open class BenchmarkExample {

    private lateinit var data: Array<Vector3d>
    private lateinit var shuffledData: Array<Vector3d>
    private lateinit var flatData: DoubleArray

    @Setup(Level.Trial)
    fun setup() {
        val rnd = Random(123456789L)
        data = Array(BenchmarkRunner.arraySize) {
            Vector3d(
                rnd.nextDouble(),
                rnd.nextDouble(),
                rnd.nextDouble()
            )
        }
        // Sort by X to shuffle them in memory
        data.sortWith(compareBy { it.x })

        shuffledData = data.sortedBy { it.x }.toTypedArray()

        flatData = data.flatMap { listOf(it.x, it.y, it.z) }.toDoubleArray()
    }


    @Benchmark
    fun naiveData(blackhole: Blackhole) {
        blackhole.consume(data.maxBy { it.y }.y)
    }

    @Benchmark
    fun naiveShuffled(blackhole: Blackhole) {
        blackhole.consume(shuffledData.maxBy { it.y }.y)
    }


    @Benchmark
    fun simpleData(blackhole: Blackhole) {
        var maxY = Double.NEGATIVE_INFINITY
        for (v in data) {
            maxY = maxOf(maxY, v.y)
        }
        blackhole.consume(maxY)
    }

    @Benchmark
    fun simpleShuffled(blackhole: Blackhole) {
        var maxY = Double.NEGATIVE_INFINITY
        for (v in shuffledData) {
            maxY = maxOf(maxY, v.y)
        }
        blackhole.consume(maxY)
    }

    @Benchmark
    fun simpleFlat(blackhole: Blackhole) {
        var maxY = Double.NEGATIVE_INFINITY
        for (i in 1 until flatData.size step 3) {
            maxY = maxOf(maxY, flatData[i])
        }
        blackhole.consume(maxY)
    }


    @Benchmark
    fun interpreterData(blackhole: Blackhole) {
        val program = findMaxProgram()
        val interpreter = Interpreter()
        interpreter.arr = data
        interpreter.execute(program)
        blackhole.consume(interpreter.maxY)
    }

    @Benchmark
    fun interpreterShuffled(blackhole: Blackhole) {
        val program = findMaxProgram()
        val interpreter = Interpreter()
        interpreter.arr = shuffledData
        interpreter.execute(program)
        blackhole.consume(interpreter.maxY)
    }

    @Benchmark
    fun interpreterFlat(blackhole: Blackhole) {
        val program = findFlatMaxProgram()
        val interpreter = Interpreter()
        interpreter.flatArr = flatData
        interpreter.execute(program)
        blackhole.consume(interpreter.maxY)
    }


    @Benchmark
    fun staticInterpreterData(blackhole: Blackhole) {
        blackhole.consume(Interpreter.executeStatic(findMaxProgram(), data))
    }

    @Benchmark
    fun staticInterpreterShuffled(blackhole: Blackhole) {
        blackhole.consume(Interpreter.executeStatic(findMaxProgram(), shuffledData))
    }

    @Benchmark
    fun staticInterpreterFlat(blackhole: Blackhole) {
        blackhole.consume(Interpreter.executeStatic(findFlatMaxProgram(), flatData))
    }


    @Benchmark
    fun virtualInterpreter(blackhole: Blackhole) {
        val program = findMaxProgramVirt()
        val interpreter = Interpreter()
        interpreter.arr = data
        interpreter.execute(program)
        blackhole.consume(interpreter.maxY)
    }

    @Benchmark
    fun virtualInterpreterShuffled(blackhole: Blackhole) {
        val program = findMaxProgramVirt()
        val interpreter = Interpreter()
        interpreter.arr = shuffledData
        interpreter.execute(program)
        blackhole.consume(interpreter.maxY)
    }

    @Benchmark
    fun virtualInterpreterFlat(blackhole: Blackhole) {
        val program = findFlatMaxProgramVirt()
        val interpreter = Interpreter()
        interpreter.flatArr = flatData
        interpreter.execute(program)
        blackhole.consume(interpreter.maxY)
    }
}


object BenchmarkRunner {
    // Vector3d contains 3 doubles, fileds size 24 bytes, each object may consume about 32
    const val arraySize: Int = 4194304

    @JvmStatic
    fun main(args: Array<String>) {
        val include = BenchmarkExample::class.java.simpleName
        val options = org.openjdk.jmh.runner.options.OptionsBuilder()
            .include(".*$include.*")
            .forks(1)
            .warmupIterations(2)
            .measurementIterations(3)
            .build()

        org.openjdk.jmh.runner.Runner(options).run()
    }
}