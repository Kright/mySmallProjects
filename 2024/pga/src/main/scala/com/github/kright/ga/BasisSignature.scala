package com.github.kright.ga

case class BasisSignature(pos: Int,
                          neg: Int,
                          zeros: Int):
  val vectorsCount: Int = pos + neg + zeros
  val bladesCount: Int = 1 << vectorsCount
  val bitsMap: Int = (1 << vectorsCount) - 1
