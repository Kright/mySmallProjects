package com.github.kright.ga

class BasisNames(str: String):
  require(str.distinct.length == str.length)
  
  val names: IndexedSeq[Char] = str
  
  def size: Int = names.length
  
  private val namesToIndices = names.zipWithIndex.toMap
  def apply(name: Char): Int = namesToIndices(name)
