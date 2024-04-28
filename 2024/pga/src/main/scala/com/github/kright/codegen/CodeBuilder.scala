package com.github.kright.codegen


class CodeBuilder(var level: Int = 0):
  val code = StringBuilder()

  def apply(body: CodeBuilder ?=> Unit): String =
    given b: CodeBuilder = this

    body
    code.toString()

object CodeBuilder:
  def code(code: String*)(using codeBuilder: CodeBuilder): Unit =
    code.foreach { line =>
      codeBuilder.code.append("  " * codeBuilder.level).append(line).append("\n")
    }

  def block(body: => Unit)(using codeBuilder: CodeBuilder): Unit =
    codeBuilder.level += 1
    body
    codeBuilder.level -= 1
