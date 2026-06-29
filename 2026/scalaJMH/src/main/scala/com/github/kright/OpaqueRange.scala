package com.github.kright

object OpaqueRange:
  opaque type OpaqueRange = Int

  inline def apply(endExclusive: Int): OpaqueRange = endExclusive

  extension (r: OpaqueRange)
    inline def foreach(inline body: Int => Unit): Unit =
      var i = 0
      while (i < r) {
        body(i)
        i += 1
      }

  opaque type OpaqueRangeWithStart = Long

  inline def apply(start: Int, endExclusive: Int): OpaqueRangeWithStart =
    (endExclusive.toLong << 32) | (start.toLong & 0xFFFFFFFFL)

  extension (r: OpaqueRangeWithStart)
    inline def start: Int = r.toInt
    inline def endExclusive: Int = (r >> 32).toInt

    inline def foreach(inline body: Int => Unit): Unit =
      var i = r.start
      val end = r.endExclusive
      while (i < end) {
        body(i)
        i += 1
      }

  extension (t: Int)
    inline infix def until(v: Int): OpaqueRangeWithStart = OpaqueRange(t, v)
    inline infix def to(v: Int): OpaqueRangeWithStart = OpaqueRange(t, v + 1)

  extension (inline zero: 0)
    inline infix def until(v: Int): OpaqueRange = OpaqueRange(v)
    inline infix def to(v: Int): OpaqueRange = OpaqueRange(v + 1)
