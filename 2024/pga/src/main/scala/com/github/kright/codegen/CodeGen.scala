package com.github.kright.codegen

import com.github.kright.codegen.CodeBuilder.*
import com.github.kright.ga.{GA, MultiVector, ga}
import com.github.kright.symbolic.SymbolicStr.given
import com.github.kright.symbolic.{SymbolicStr, SymbolicToPrettyString}

/*
@main
def generatePga3() = GA.pga3.use {
  val bladesOrder = ga.bladesOrderedByGrade.all

  println(generateBinOp(
    "geometric",
    Seq("⟑"),
    MultivectorType.multivector,
    MultivectorType.multivector,
    Option("PGA3IMultivector"),
    Option("PGA3IMultivector"),
    (a, b) => a.geometric(b),
  ))

  println(generateBinOp(
    "dot",
    Seq("⋅"),
    MultivectorType.multivector,
    MultivectorType.multivector,
    Option("PGA3IMultivector"),
    Option("PGA3IMultivector"),
    (a, b) => a.dot(b),
  ))

  println(generateBinOp(
    "wedge",
    Seq("^", "∧"),
    MultivectorType.multivector,
    MultivectorType.multivector,
    Option("PGA3IMultivector"),
    Option("PGA3IMultivector"),
    (a, b) => a.wedge(b),
  ))

  println(generateOp(
    "reverse",
    Seq("~"),
    MultivectorType.multivector,
    Option("PGA3IMultivector"),
    a => a.reverse,
  ))

  println(generateOp(
    "rightComplement",
    Seq(),
    MultivectorType.multivector,
    Option("PGA3IMultivector"),
    a => a.rightComplement,
  ))

  println(generateOp(
    "leftComplement",
    Seq(),
    MultivectorType.multivector,
    Option("PGA3IMultivector"),
    a => a.leftComplement,
  ))
}


private def generateBinOp(binopName: String,
                          aliases: Seq[String],
                          a: MultivectorType,
                          b: MultivectorType,
                          overrideInputNameA: Option[String] = None,
                          overrideInputNameB: Option[String] = None,
                          func: (MultiVector[SymbolicStr], MultiVector[SymbolicStr]) => MultiVector[SymbolicStr])
                         (using basis: GA): String = {
  val as: MultiVector[SymbolicStr] = a.makeSymbolic("a")
  val bs: MultiVector[SymbolicStr] = b.makeSymbolic("b")

  val result: MultiVector[String] =
    func(as, bs)
      .simplified
      .mapValues(SymbolicToPrettyString(_))
      .mapValues(s => if (s.startsWith("(") && s.endsWith(")")) s.substring(1, s.length - 1) else s)

  val resultType = MultivectorType.findType(result)

  val aNameInp = overrideInputNameA.getOrElse(a.name)
  val bNameInp = overrideInputNameB.getOrElse(b.name)

  CodeBuilder(1) {
    code(s"extension (a: ${aNameInp})")
    block {
      code(
        "@static",
        s"@targetName(\"${binopName}_${aNameInp}_${b.name}\")",
        s"infix def ${binopName}(b: ${bNameInp}): ${resultType.name} ="
      )
      code(s"${resultType.name}(")
      block {
        resultType.usedBlades.foreach { blade =>
          code(s"${blade} = ${result.values.getOrElse(blade, "0.0")},")
        }
      }
      code(
        s")",
        "",
      )

      aliases.foreach { alias =>
        code(
          s"inline infix def ${alias}(b: ${bNameInp}): ${resultType.name} = a.${binopName}(b)",
          ""
        )
      }
    }
  }
}


private def generateOp(opName: String,
                       aliases: Seq[String],
                       a: MultivectorType,
                       overrideInputNameA: Option[String] = None,
                       func: MultiVector[SymbolicStr] => MultiVector[SymbolicStr])
                      (using basis: GA): String = {
  val aNameInp = overrideInputNameA.getOrElse(a.name)
  val as: MultiVector[SymbolicStr] = a.makeSymbolic("a")

  val result: MultiVector[String] =
    func(as)
      .simplified
      .mapValues(SymbolicToPrettyString(_))
      .mapValues(s => if (s.startsWith("(") && s.endsWith(")")) s.substring(1, s.length - 1) else s)

  val resultType = MultivectorType.findType(result)

  CodeBuilder(1) {
    code(s"extension (a: ${aNameInp})")
    block {
      code(
        "@static",
        s"@targetName(\"${opName}_${aNameInp}\")",
        s"def ${opName}: ${resultType.name} ="
      )
      code(s"${resultType.name}(")
      block {
        resultType.usedBlades.foreach { blade =>
          code(s"${blade} = ${result.values.getOrElse(blade, "0.0")},")
        }
      }
      code(
        s")",
        "",
      )
      aliases.foreach { alias =>
        code(
          s"inline def ${alias} : ${resultType.name} = a.${opName}",
          ""
        )
      }
    }
  }
}

 */