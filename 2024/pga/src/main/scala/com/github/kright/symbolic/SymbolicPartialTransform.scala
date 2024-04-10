package com.github.kright.symbolic

trait SymbolicPartialTransform[F, S] extends (Symbolic[F, S] => Option[Symbolic[F, S]]):
  self =>

  def apply(elems: Seq[Symbolic[F, S]]): Option[Seq[Symbolic[F, S]]] = {
    val newElems = elems.map(apply)
    if (newElems.forall(_.isEmpty)) return None
    Option(newElems.zip(elems).map((next, prev) => next.getOrElse(prev)))
  }

  def asSymbolicTransform: SymbolicTransform[F, S, F, S] =
    (s: Symbolic[F, S]) => self(s).getOrElse(s)

  def repeat(maxRepeatCount: Int): SymbolicTransformRepeater[F, S] =
    SymbolicTransformRepeater(this, maxRepeatCount)

  def withDebugLog(prefix: String): SymbolicPartialTransform[F, S] =
    (s: Symbolic[F, S]) => {
      val r = self(s)
      println(s"$prefix: $s => $r")
      r
    }
