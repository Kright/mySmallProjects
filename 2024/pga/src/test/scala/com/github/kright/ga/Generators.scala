package com.github.kright.ga

import org.scalacheck.Gen

import java.security.SecureRandom

object Generators:
  private val rnd = SecureRandom()

  def allBasisesSeq =
    IndexedSeq(Basis.ga2, Basis.ga3, Basis.ga4, Basis.pga2, Basis.pga3)

  def allBasises: Gen[Basis] =
    Gen.oneOf(allBasisesSeq)

  def forAnyBasis(body: Basis ?=> Unit): Unit = {
    for (basis <- allBasisesSeq) {
      basis.use {
        body
      }
    }
  }  

  extension (basis: Basis)
    def basisBladesGen: Gen[BasisBlade] =
      Gen.oneOf(basis.blades)

    def basisMultivectorsGen: Gen[MultiVector[Double]] =
      basisBladesGen.map(MultiVector(_)(using basis))

    def multivectorsGen: Gen[MultiVector[Double]] =
      Gen.containerOfN[Seq, Double](basis.bladesCount, Gen.double).map { values =>
        require(values.length == basis.bladesCount)
        MultiVector(basis.blades.zip(values))(using basis)
      }

    def bladesGen(order: Int): Gen[MultiVector[Double]] =
      multivectorsGen.map(_.filter((b, v) => b.order == order))
