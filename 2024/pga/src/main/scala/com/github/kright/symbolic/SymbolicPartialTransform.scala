package com.github.kright.symbolic

trait SymbolicPartialTransform extends (Symbolic => Option[Symbolic]):
  self =>

  def apply(elems: Seq[Symbolic]): Option[Seq[Symbolic]] = {
    val newElems = elems.map(apply)
    if (newElems.forall(_.isEmpty)) return None
    Option(newElems.zip(elems).map((next, prev) => next.getOrElse(prev)))
  }

  def asSymbolicTransform: SymbolicTransform =
    (s: Symbolic) => self(s).getOrElse(s)

  def repeat(maxRepeatCount: Int): SymbolicTransformRepeater =
    SymbolicTransformRepeater(this, maxRepeatCount)

  def withDebugLog(prefix: String): SymbolicPartialTransform =
    (s: Symbolic) => {
      val r = self(s)
      println(s"$prefix: $s => $r")
      r
    }
