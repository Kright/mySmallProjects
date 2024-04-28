package com.github.kright.codegen

import com.github.kright.ga.{GA, BasisBlade, PGA3, MultiVector}
import com.github.kright.symbolic.SymbolicStr

//case class MultivectorType(name: String,
//                           usedBlades: Seq[BasisBlade]):
//  private val bladesSet = usedBlades.toSet
//
//  def makeSymbolic(baseName: String)(using GA) =
//    MultiVector(usedBlades.map(blade => blade -> SymbolicStr(s"${baseName}.${blade}")).toMap)
//
//  def contains(v: MultiVector[?]): Boolean =
//    v.values.keys.forall(bladesSet.contains)
//
//object MultivectorType:
//  private given b: PGA3 = GA.pga3
//
//  val scalar = MultivectorType("PGA3Scalar", b.bladesOrderedByGrade.all.take(1))
//  val plane = MultivectorType("PGA3Plane", b.bladesOrderedByGrade.all.filter(_.grade == 1))
//  val multivector = MultivectorType("PGA3Multivector", b.bladesOrderedByGrade.all)
//
//  val multivectors: Seq[MultivectorType] =
//    Seq(
//      scalar,
//      plane,
//      multivector
//    )
//
//  def findType(v: MultiVector[?]): MultivectorType =
//    multivectors.find(_.contains(v)).get
