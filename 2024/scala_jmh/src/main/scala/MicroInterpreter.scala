package com.github.kright

trait MicroInterpreter:
  def apply(stack: IntStack): Unit

final case class PushConst(value: Int) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit = stack.push(value)

case object DecHead extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    stack.head -= 1

final case class Dup(shift: Int) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    stack.dup(shift)

final case class Call(m: () => MicroInterpreter) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    m()(stack)

case object Add extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    stack.add()

final case class CopyHead(shift: Int) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    stack.copyHead(shift)

private final case class SetHead(value: Int) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    stack.head = value

case object Pop extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    stack.pop()

final case class InstructionsSequence(instructions: Array[MicroInterpreter]) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    for (instruction <- instructions) {
      instruction(stack)
    }

object OptimizedInstructionSequence:
  def apply(instr: MicroInterpreter*): MicroInterpreter =
    require(instr.nonEmpty)

    instr.size match
      case 1 => instr.head
      case 2 => InstructionsSequence2(instr(0), instr(1))
      case 3 => InstructionsSequence3(instr(0), instr(1), instr(2))
      case 4 => InstructionsSequence4(instr(0), instr(1), instr(2), instr(3))
      case 5 => InstructionsSequence5(instr(0), instr(1), instr(2), instr(3), instr(4))
      case _ => InstructionsSequence5(instr(0), instr(1), instr(2), instr(3), apply(instr.drop(4) *))

final case class InstructionsSequence2(m0: MicroInterpreter, m1: MicroInterpreter) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    m0(stack)
    m1(stack)

final case class InstructionsSequence3(m0: MicroInterpreter, m1: MicroInterpreter, m2: MicroInterpreter) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    m0(stack)
    m1(stack)
    m2(stack)

final case class InstructionsSequence4(m0: MicroInterpreter,
                                       m1: MicroInterpreter,
                                       m2: MicroInterpreter,
                                       m3: MicroInterpreter) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    m0(stack)
    m1(stack)
    m2(stack)
    m3(stack)

final case class InstructionsSequence5(m0: MicroInterpreter,
                                       m1: MicroInterpreter,
                                       m2: MicroInterpreter,
                                       m3: MicroInterpreter,
                                       m4: MicroInterpreter) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    m0(stack)
    m1(stack)
    m2(stack)
    m3(stack)
    m4(stack)

final case class InstructionsSequence8(m0: MicroInterpreter,
                                       m1: MicroInterpreter,
                                       m2: MicroInterpreter,
                                       m3: MicroInterpreter,
                                       m4: MicroInterpreter,
                                       m5: MicroInterpreter,
                                       m6: MicroInterpreter,
                                       m7: MicroInterpreter) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    m0(stack)
    m1(stack)
    m2(stack)
    m3(stack)
    m4(stack)
    m5(stack)
    m6(stack)
    m7(stack)

final case class IfHeadLE(bodyLE: MicroInterpreter, bodyB: MicroInterpreter) extends MicroInterpreter:
  override def apply(stack: IntStack): Unit =
    if (stack.head <= 0) bodyLE(stack)
    else bodyB(stack)
