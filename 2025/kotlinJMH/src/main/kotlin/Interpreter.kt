package org.example

class Interpreter {
    var pos = 0

    var i: Int = 0
    var arrSize: Int = 0

    var maxY: Double = Double.NEGATIVE_INFINITY
    var y: Double = 0.0

    var arr: Array<Vector3d>? = null
    var flatArr: DoubleArray? = null
    var current: Vector3d? = null

    fun execute(instruction: Array<VirtualInstruction>) {
        while (true) {
            val next = instruction[pos++]
            if (!next.execute(this)) {
                return
            }
        }
    }

    fun execute(instructions: IntArray) {
        while (true) {
            val next = instructions[pos++]

            when (next) {
                Instruction.GET_ARR_SIZE -> arrSize = arr!!.size
                Instruction.CHECK_I_SIZE -> if (i >= arrSize) return
                Instruction.GET_VEC ->  current = arr!![i]
                Instruction.GET_Y -> y = current!!.y
                Instruction.UPDATE_MAX -> maxY = maxOf(maxY, y)
                Instruction.INC_I -> i += 1
                Instruction.INC2_I ->  i += 2
                Instruction.INC3_I ->  i += 3
                Instruction.JUMP1 ->  pos = 1
                Instruction.JUMP2 -> pos = 2
                Instruction.GET_DOUBLE_FROM_ARR -> y = flatArr!![i]
                Instruction.GET_DOUBLE_ARR_SIZE -> arrSize = flatArr!!.size
            }
        }
    }

    companion object {
        fun executeStatic(instructions: IntArray, arr: Array<Vector3d>): Double {
            var pos = 0
            var i = 0
            var arrSize = 0 // arr length won’t change; keep local

            var maxY = Double.NEGATIVE_INFINITY
            var y = 0.0
            var current: Vector3d? = null

            while (true) {
                when (instructions[pos++]) {
                    Instruction.GET_ARR_SIZE -> arrSize = arr.size
                    Instruction.GET_DOUBLE_ARR_SIZE -> TODO("Not yet implemented")
                    Instruction.CHECK_I_SIZE -> if (i >= arrSize) return maxY
                    Instruction.GET_VEC -> current = arr[i]
                    Instruction.GET_Y -> y = current!!.y
                    Instruction.UPDATE_MAX -> maxY = if (y > maxY) y else maxY
                    Instruction.INC_I -> i += 1
                    Instruction.INC2_I -> i += 2
                    Instruction.INC3_I -> i += 3
                    Instruction.JUMP1 -> pos = 1
                    Instruction.JUMP2 -> pos = 2
                }
            }
        }

        fun executeStatic(instructions: IntArray, arr: DoubleArray): Double {
            var pos = 0
            var i = 0
            var arrSize = 0 // arr length won’t change; keep local

            var maxY = Double.NEGATIVE_INFINITY
            var y = 0.0

            while (true) {
                when (instructions[pos++]) {
                    Instruction.GET_DOUBLE_ARR_SIZE -> arrSize = arr.size
                    Instruction.CHECK_I_SIZE -> if (i >= arrSize) return maxY
                    Instruction.UPDATE_MAX -> maxY = if (y > maxY) y else maxY
                    Instruction.INC_I -> i += 1
                    Instruction.INC2_I -> i += 2
                    Instruction.INC3_I -> i += 3
                    Instruction.JUMP1 -> pos = 1
                    Instruction.JUMP2 -> pos = 2
                    Instruction.GET_DOUBLE_FROM_ARR -> y = arr[i]
                }
            }
        }
    }
}

fun findMaxProgram(): IntArray {
    return intArrayOf(
        Instruction.GET_ARR_SIZE,
        Instruction.CHECK_I_SIZE,
        Instruction.GET_VEC,
        Instruction.GET_Y,
        Instruction.UPDATE_MAX,
        Instruction.INC_I,
        Instruction.JUMP1,
    )
}

fun findFlatMaxProgram(): IntArray {
    return intArrayOf(
        Instruction.GET_DOUBLE_ARR_SIZE,
        Instruction.INC_I,
        Instruction.CHECK_I_SIZE,
        Instruction.GET_DOUBLE_FROM_ARR,
        Instruction.UPDATE_MAX,
        Instruction.INC3_I,
        Instruction.JUMP2,
    )
}


fun findMaxProgramVirt(): Array<VirtualInstruction> {
    return arrayOf(
        VirtualInstruction.GetArrSize,
        VirtualInstruction.CheckISize,
        VirtualInstruction.Block5(
            VirtualInstruction.GetVec,
            VirtualInstruction.GetY,
            VirtualInstruction.UpdateMax,
            VirtualInstruction.IncI(1),
            VirtualInstruction.Jump(1)
        ),
    )
}

fun findFlatMaxProgramVirt(): Array<VirtualInstruction> {
    return arrayOf(
        VirtualInstruction.GetDoubleArrSize,
        VirtualInstruction.IncI(1),
        VirtualInstruction.CheckISize,
        VirtualInstruction.Block4(
            VirtualInstruction.GetDoubleArrValue,
            VirtualInstruction.UpdateMax,
            VirtualInstruction.IncI(3),
            VirtualInstruction.Jump(2)
        ),
    )
}



object Instruction {
    const val GET_ARR_SIZE: Int = 1
    const val GET_DOUBLE_ARR_SIZE: Int = 2

    const val CHECK_I_SIZE: Int = 3
    const val GET_VEC: Int = 4
    const val GET_Y: Int = 5
    const val UPDATE_MAX: Int = 6
    const val INC_I: Int = 7
    const val INC2_I: Int = 8
    const val INC3_I: Int = 9
    const val JUMP1: Int = 10
    const val JUMP2: Int = 11
    const val GET_DOUBLE_FROM_ARR: Int = 12
}
