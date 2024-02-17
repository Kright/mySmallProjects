package com.github.kright.ga

import scala.collection.mutable

class MultiplicationRule(using basis: Basis) extends HasBasis(basis):
  def dot(left: BasisBlade, right: BasisBlade): (BasisBlade, Sign) =
    if (left.hasCommonBasisVectors(right) || left == right) {
      geometric(left, right)
    } else {
      (basis.scalarBlade, Sign.Zero)
    }

  def wedge(left: BasisBlade, right: BasisBlade): (BasisBlade, Sign) =
    if (left.hasCommonBasisVectors(right) || left == right) {
      (basis.scalarBlade, Sign.Zero)
    } else {
      geometric(left, right)
    }

  def geometric(a: BasisBlade, b: BasisBlade): (BasisBlade, Sign) =
    checkBasis(a, b)

    val allBasisVectors = a.basisVectors ++ b.basisVectors
    val sorted = allBasisVectors.sorted

    val paritySign = if (Permutation.parity(allBasisVectors, sorted)) Sign.Positive else Sign.Negative
    val removedSigns = a.commonBasisVectors(b).map(_.getSquareSign)
    val sign = removedSigns.fold(paritySign)(_ * _)

    if (sign == Sign.Zero) return (basis.scalarBlade, sign)
    (BasisBlade(a.bits ^ b.bits), sign)

  def wedge[V: MyAlgebra](left: MultiVector[V], right: MultiVector[V]): MultiVector[V] = {
    checkBasis(left, right)

    val result = new mutable.HashMap[BasisBlade, V]()

    for ((leftBlade, leftV) <- left.values) {
      for ((rightBlade, rightV) <- right.values) {
        val (basis, sign) = wedge(leftBlade, rightBlade)
        if (sign != Sign.Zero) {
          var mult: V = leftV * rightV
          if (sign == Sign.Negative) {
            mult = -mult
          }

          result(basis) =
            if (result.contains(basis)) result(basis) + mult
            else mult
        }
      }
    }

    MultiVector(result.toMap)
  }

  private def checkBasis(left: HasBasis, right: HasBasis): Unit = {
    require(left.basis == basis)
    require(right.basis == basis)
  }
