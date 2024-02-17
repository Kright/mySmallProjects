package com.github.kright.ga

trait MyAlgebra[T]:
  
  extension (a: T)
    def +(b: T): T
    def *(b: T): T
    def unary_- : T
    
//  def plus(a: T, b: T): T
//  def multiply(a: T, b: T): T
//  def multiplyToDouble(a: T, b: Double): T

object MyAlgebra:
  given MyAlgebra[Double] with
    extension (a: Double) 
      override def +(b: Double): Double = a + b
      override def *(b: Double): Double = a * b
      override def unary_- : Double = -a


