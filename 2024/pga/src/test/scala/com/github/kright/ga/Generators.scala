package com.github.kright.ga

import org.scalacheck.Gen

import java.security.SecureRandom

object Generators:
  private val rnd = SecureRandom()

  def allBasisesSeq =
    IndexedSeq(Basis.ga2, Basis.ga3, Basis.ga4, Basis.pga2, Basis.pga3)

  def allBasises: Gen[Basis] =
    Gen.oneOf(allBasisesSeq)

  extension (basis: Basis)
    def basisBladesGen: Gen[BasisBlade] =
      Gen.oneOf(basis.blades)

    def basisMultivectorsGen: Gen[MultiVector[Double]] =
      basisBladesGen.map(MultiVector(_)(using basis))

    def multivectorsGen: Gen[MultiVector[Double]] =
      for {
        order <- Gen.choose(1, basis.vectorsCount)
        components <- Gen.containerOfN[Seq, MultiVector[Double]](basis.bladesCount, bladesGen(1))
      } yield components.reduce(_ ⟑ _)

    def bladesGen(order: Int): Gen[MultiVector[Double]] =
      Gen.containerOfN[Seq, Double](basis.bladesCount, Gen.double).map { values =>
        require(values.length == basis.bladesCount)
        MultiVector(basis.blades.zip(values).filter((b, v) => b.order == order))(using basis)
      }
