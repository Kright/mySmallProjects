package org.example

sealed interface VirtualInstruction {
    fun execute(interpreter: Interpreter): Boolean

    object GetArrSize : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            interpreter.arrSize = interpreter.arr!!.size
            return true
        }
    }

    object GetDoubleArrSize : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            interpreter.arrSize = interpreter.flatArr!!.size
            return true
        }
    }

    object GetDoubleArrValue : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            interpreter.y = interpreter.flatArr!![interpreter.i]
            return true
        }
    }

    object CheckISize : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            return interpreter.i < interpreter.arrSize
        }
    }

    object GetVec : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            interpreter.current = interpreter.arr!![interpreter.i]
            return true
        }
    }

    object GetY : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            interpreter.y = interpreter.current!!.y
            return true
        }
    }

    object UpdateMax : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            interpreter.maxY = maxOf(interpreter.maxY, interpreter.y)
            return true
        }
    }

    class IncI(val value: Int) : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            interpreter.i += value
            return true
        }
    }

    class Jump(val pos: Int) : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            interpreter.pos = pos
            return true
        }
    }

    class Block4(
        val i0: VirtualInstruction,
        val i1: VirtualInstruction,
        val i2: VirtualInstruction,
        val i3: VirtualInstruction,
    ) : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            i0.execute(interpreter)
            i1.execute(interpreter)
            i2.execute(interpreter)
            i3.execute(interpreter)
            return true
        }
    }

    class Block5(
        val i0: VirtualInstruction,
        val i1: VirtualInstruction,
        val i2: VirtualInstruction,
        val i3: VirtualInstruction,
        val i4: VirtualInstruction,
    ) : VirtualInstruction {
        override fun execute(interpreter: Interpreter): Boolean {
            i0.execute(interpreter)
            i1.execute(interpreter)
            i2.execute(interpreter)
            i3.execute(interpreter)
            i4.execute(interpreter)
            return true
        }
    }
}