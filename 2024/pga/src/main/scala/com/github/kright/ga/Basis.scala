package com.github.kright.ga

case class Basis(pos: Int,
                 neg: Int,
                 zeros: Int,
                 basisNames: BasisNames):
  private given self: Basis = this

  val vectorsCount: Int = pos + neg + zeros
  val vectors: IndexedSeq[BasisVector] = (0 until vectorsCount).map(BasisVector(_))

  val bladesCount: Int = 1 << vectorsCount
  val blades: IndexedSeq[BasisBlade] = (0 until bladesCount).map(b => BasisBlade(b))
  val bitsMap: Int = (1 << vectorsCount) - 1

  def scalarBlade: BasisBlade = blades(0)
  def antiScalarBlade: BasisBlade = bladesByOrder.last

  val bladesByOrder: IndexedSeq[BasisBlade] = blades.sortBy(_.order)
  require(basisNames.size == vectorsCount)

  private val rule = MultiplicationRules()

  val geometric = MultiplicationTable(rule.geometric)
  val wedge = MultiplicationTable(rule.wedge)
  val dot = MultiplicationTable(rule.dot)
  
  val geometricAntiproduct = MultiplicationTable(rule.geometricAntiproduct)
  val wedgeAntiproduct = MultiplicationTable(rule.wedgeAntiproduct)
  val dotAntiproduct = MultiplicationTable(rule.dotAntiproduct)
  
  val leftComplement = SingleOpTable(rule.leftComplement)
  val rightComplement = SingleOpTable(rule.rightComplement)
  val bulk = SingleOpTable(rule.bulk)
  val weight = SingleOpTable(rule.weight)
  val reverse = SingleOpTable(rule.reverse)
  val antiReverse = SingleOpTable(rule.antiReverse)

  override def equals(obj: Any): Boolean =
    if (this eq obj.asInstanceOf[Object]) return true
    obj match {
      case Basis(this.pos, this.neg, this.zeros, _) => true
      case _ => false
    }

  override def hashCode(): Int =
    scala.util.hashing.byteswap32((pos << 16) + (neg << 8) + zeros)

  def use[T](f: Basis ?=> T): T =
    given basis: Basis = this

    f


def basis(using b: Basis): Basis = b

object Basis:
  val ga2: Basis = Basis(2, 0, 0, BasisNames("xy"))
  val ga3: Basis = Basis(3, 0, 0, BasisNames("xyz"))
  val ga4: Basis = Basis(4, 0, 0, BasisNames("xyzw"))

  // projective geometric algebra
  val pga2: Basis = Basis(2, 0, 1, BasisNames("xyw"))
  val pga3: Basis = Basis(3, 0, 1, BasisNames("xyzw"))


trait HasBasis(val basis: Basis)