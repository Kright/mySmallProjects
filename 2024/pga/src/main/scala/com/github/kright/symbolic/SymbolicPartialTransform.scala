package com.github.kright.symbolic

trait SymbolicPartialTransform extends (SimpleSymbolic => Option[SimpleSymbolic]):
  self =>

  def apply(elems: Seq[SimpleSymbolic]): Option[Seq[SimpleSymbolic]] = {
    val newElems = elems.map(apply)
    if (newElems.forall(_.isEmpty)) return None
    Option(newElems.zip(elems).map((next, prev) => next.getOrElse(prev)))
  }

  def asSymbolicTransform: SymbolicTransform =
    (s: SimpleSymbolic) => self(s).getOrElse(s)

  def repeat(maxRepeatCount: Int): SymbolicTransformRepeater =
    SymbolicTransformRepeater(this, maxRepeatCount)

  def withDebugLog(prefix: String): SymbolicPartialTransform =
    (s: SimpleSymbolic) => {
      val r = self(s)
      println(s"$prefix: $s => $r")
      r
    }
