package com.github.kright

import com.github.kright.Utils.MyStringExt
import com.github.kright.Rule.{consumeSuffix, hasSuffix}

import scala.util.matching.Regex

object PotterStem {
  def apply(word: String): String = {
    val w = word.toLowerCase.replace("ё", "е")
    val rvw = rv(w)
    (w - rvw) + stem(rvw).get
  }

  private lazy val stem = step1 |> step2 |> step3 |> step4

  private def step1: Rule = rmPerfectiveGerund | rmReflexive | rmAdjectival | rmVerb | rmNoun | Rule.ok
  private def step2: Rule = consumeSuffix(Seq("и")) | Rule.ok
  private def step3: Rule = rmDerivational | Rule.ok
  private def step4: Rule =
    rmNifNN |
      (rmSuperlative |> (rmNifNN | Rule.ok)) |
      consumeSuffix(Seq("ь")) |
      Rule.ok

  private def rmNifNN: Rule = hasSuffix(Seq("нн")) |> consumeSuffix(Seq("н"))

  private def rmPerfectiveGerund: Rule =
    (consumeSuffix(Seq("в", "вши", "вшись")) |> hasSuffix(Seq("а", "я"))) |
      consumeSuffix(Seq("ив", "ивши", "ившись", "ыв", "ывши", "ывшись"))

  private def rmReflexive: Rule = consumeSuffix(Seq("ся", "сь"))

  private def adjective: Array[String] = "ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею".split(Regex.quote("|"))
  private def participleG1: Seq[String]= Seq("ем", "нн", "вш", "ющ", "щ")
  private def participleG2: Seq[String]= Seq("ивш", "ывш", "ующ")

  private def rmAdjectival: Rule = Rule{ word =>
    val rvw = rv(word)
    var maxSuffixLen = 0

    for (adj <- adjective) {
      if (rvw.endsWith(adj)) {
        maxSuffixLen = math.max(maxSuffixLen, adj.size)

        val p = rvw - adj

        for(g1 <- participleG1) {
          if (p.endsWith("а" + g1) | p.endsWith("я" + g1)) {
            maxSuffixLen = math.max(maxSuffixLen, g1.size + adj.size)
          }
        }

        for(g2 <- participleG2) {
          if (p.endsWith(g2)) {
            maxSuffixLen = math.max(maxSuffixLen, g2.size + adj.size)
          }
        }
      }
    }

    if (maxSuffixLen > 0) {
      Option(word.substring(0, word.size - maxSuffixLen))
    }
    else None
  }

  private def rmVerb: Rule =
    (consumeSuffix(Seq("ла", "на", "ете", "йте", "ли", "й", "л", "ем", "н", "ло","но","ет", "ют","ны","ть","ешь","нно")) |> hasSuffix(Seq("а", "я")))|
      consumeSuffix("ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю".split(Regex.quote("|")))

  private def rmNoun: Rule =
    consumeSuffix("а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я".split(Regex.quote("|")))

  private def derivational = Seq("ост", "ость")

  private def rmDerivational: Rule = Rule{ word =>
    hasSuffix(derivational)(r2AfterRv(word)).map(_ => consumeSuffix(derivational)(word).get)
  }

  private def rmSuperlative: Rule = consumeSuffix(Seq("ейш", "ейше"))

  private def glasnie = "аеиоуыэюя"

  private def rv(word: String): String = {
    word.zipWithIndex
      .find{ case (c, _) => glasnie.contains(c) }
      .map{ case (_, pos) => word.substring(pos + 1)}
      .getOrElse("")
  }

  private def r1(word: String): String =  {
    word.zipWithIndex.drop(1).find{ case (c2, pos) =>
      val c1 = word(pos - 1)
      glasnie.contains(c1) && !glasnie.contains(c2)
    }.map{ case (_, pos) =>
      word.substring(pos + 1)
    }.getOrElse("")
  }

  private def r2(word: String): String = r1(r1(word))

  private def r2AfterRv(word: String): String = r2("a" + word)
}


private trait Rule extends (String => Option[String]){
  def |(another: Rule): Rule = Rule(s => this(s).orElse(another(s)))

  def |>(next: Rule): Rule = Rule(s => this(s).flatMap(next))
}

private object Rule {
  def apply(f: String => Option[String]): Rule = (w: String) => f(w)

  def consumeSuffix(suffixes: Iterable[String]): Rule = Rule {word =>
    suffixes.find(suffix => word.endsWith(suffix)).map(suffix => word - suffix)
  }

  def hasSuffix(suffixes: Iterable[String]): Rule = Rule { word =>
    suffixes.find(suffix => word.endsWith(suffix)).map(_ => word)
  }

  val ok: Rule = Rule(w => Option(w))

  val fail: Rule = Rule(_ => None)
}


