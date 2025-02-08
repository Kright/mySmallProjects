package com.github.kright


import com.github.kright.Utils.MyStringExt

trait Replier {
  def apply(text: String): (Option[String], String)
}

object Replier {
  private def debug(msg: String) = println(msg)

  def load(path: String): Replier = {
    val msgs = TextMessage.fromFile(path)
    val msgsByIds = msgs.map(msg => msg.id -> msg).toMap

    val tokensByIds: Map[Int, Seq[String]] = msgsByIds.map(pair => pair._1 -> BagOfWords.extract(pair._2.text))
    val termsCount = TotalFrequencyCounter.words.mapOutput(math.max(_, 1))
    termsCount.add(tokensByIds.values.flatten)

    debug(s"terms = $termsCount")

    val answersByQuestionId: Map[Int, TextMessage] =
      msgs.flatMap { msg =>
        msg.replyToMessageId.flatMap { prevId =>
          msgsByIds.get(prevId).map { prevMsg =>
            prevMsg.id -> msg
          }
        }
      }.toMap

    val fastMatchersByIds: Map[Int, FastMatcher] =
      answersByQuestionId.map { case (id, _) => id -> FastMatcher(tokensByIds(id), { t => 1.0 / termsCount(t) }) }
    val termsIndex = SearchIndex(answersByQuestionId.keys.map(id => tokensByIds(id) -> id))

    debug(s"answers count = ${answersByQuestionId.size}")

    new Replier {
      override def apply(text: String): (Option[String], String) = {
        val allQuestionTerms = BagOfWords.extract(text)
        val fastMatcher = FastMatcher(allQuestionTerms, { t => 1.0 / termsCount(t) })

        val similarityThreshold = 0.5

        val ci = termsIndex(allQuestionTerms)
        
        val cis = ci.toSeq
        
        val candidatesIds = cis.filter { id => fastMatcher.getSimilarity(fastMatchersByIds(id)) >= similarityThreshold }
        

        if (candidatesIds.isEmpty) {
          return (None, s"bestQuestions is empty, question terms = [${allQuestionTerms.mkString(", ").limitLength(100)}]")
        }

        Utils.getRandom(candidatesIds).map { questionId =>
          val answerText = answersByQuestionId(questionId).text
          val debugInfo =
            s"""
               |${text.quoted} as ${allQuestionTerms.mkString(" ").quoted(maxLength = 100)}
               |${msgsByIds(questionId).text.quoted} as ${tokensByIds(questionId).mkString(" ").quoted(maxLength = 100)}
               |reply = $answerText
               |similarity = ${fastMatcher.getSimilarity(fastMatchersByIds(questionId))}
               |questions count = ${candidatesIds.size}
               |variants = [${candidatesIds.take(10).map(id => msgsByIds(id).text.quoted(maxLength = 80) + " => " + answersByQuestionId(id).text.quoted(maxLength = 80)).mkString("\n")}]
               |""".stripMargin
          (Option(answerText), debugInfo)
        }.get
      }
    }
  }
}

