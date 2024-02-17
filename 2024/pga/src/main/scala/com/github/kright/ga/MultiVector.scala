package com.github.kright.ga

case class MultiVector[Value](values: Map[BasisBlade, Value])(using basis: Basis) extends HasBasis(basis):

  override def toString: String = s"values = ${values}"
