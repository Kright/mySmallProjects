package com.github.kright.ga

case class Basis(signature: BasisSignature,
                 basisNames: BasisNames):
  private given self: Basis = this

  val vectors: IndexedSeq[BasisVector] = (0 until signature.vectorsCount).map(BasisVector(_))

  val blades: IndexedSeq[BasisBlade] = (0 until signature.bladesCount).map(b => BasisBlade(b))

  def scalarBlade: BasisBlade = blades(0)

  def antiScalarBlade: BasisBlade = bladesByOrder.last

  val bladesByOrder: IndexedSeq[BasisBlade] = blades.sortBy(_.grade)
  require(basisNames.size == signature.vectorsCount)

  private val rule = MultiplicationRules()

  val geometric = MultiplicationTable(rule.geometric)
  val wedge = MultiplicationTable(rule.wedge)
  val dot = MultiplicationTable(rule.dot)

  val antiGeometric = MultiplicationTable(rule.antiGeometric)
  val antiWedge = MultiplicationTable(rule.antiWedge)
  val antiDot = MultiplicationTable(rule.antiDot)

  val leftComplement = SingleOpTable(rule.leftComplement)
  val rightComplement = SingleOpTable(rule.rightComplement)
  val bulk = SingleOpTable(rule.bulk)
  val weight = SingleOpTable(rule.weight)
  val reverse = SingleOpTable(rule.reverse)
  val antiReverse = SingleOpTable(rule.antiReverse)

  override def equals(obj: Any): Boolean =
    if (this eq obj.asInstanceOf[Object]) return true
    obj match {
      case Basis(otherSignature, _) => signature == otherSignature
      case _ => false
    }

  override def hashCode(): Int =
    signature.hashCode()


def basis(using b: Basis): Basis = b

object Basis:
  val ga2: Basis = Basis(BasisSignature(2, 0, 0), BasisNames("xy"))
  val ga3: Basis = Basis(BasisSignature(3, 0, 0), BasisNames("xyz"))
  val ga4: Basis = Basis(BasisSignature(4, 0, 0), BasisNames("xyzw"))

  // projective geometric algebra
  val pga2: BasisPGA2 = BasisPGA2(BasisNames("xyw"))
  val pga3: BasisPGA3 = BasisPGA3(BasisNames("xyzw"))

  extension[B <: Basis] (basis: B)
    def use[T](f: B ?=> T): T =
      given b: B = basis
      f


trait HasBasis(val basis: Basis)