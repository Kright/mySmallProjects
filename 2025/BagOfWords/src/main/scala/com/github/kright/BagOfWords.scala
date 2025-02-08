package com.github.kright

import scala.collection.mutable.ArrayBuffer

object BagOfWords {
  def extract(text: String): Seq[String] = {
    tokenize(text).map(stem).filter(_.nonEmpty).toSeq
  }

  def extractMainWords(msg: Seq[String], totalFrequencies: String => Int, mainWordsCount: Int): Seq[String] = {
    if (msg.size <= mainWordsCount) return msg
    val ww = msg.groupBy(w => w).map( p => p._1 -> p._2.size)

    ww.map{ case (word, count) =>
      word -> -(count.toDouble / math.max(totalFrequencies(word), 1))
    }.toArray.sortBy(_._2).take(mainWordsCount).map(_._1).toSeq
  }

  def getSimilarity(msg1: Seq[String], msg2: Seq[String], weights: String => Double): Double = {
    val ww1: Set[String] = msg1.toSet
    val ww2: Set[String] = msg2.toSet

    val positiveScore = ww1.intersect(ww2).toSeq.map(weights).sum
    val totalScore = (ww1 ++ ww2).toSeq.map(weights).sum

    positiveScore / totalScore
  }

  // faster version
  def getSimilarity2(ww1: Set[String], ww2: Set[String], weights: String => Double): Double = {
    var positiveScore = 0.0
    var totalScore = 0.0
    for (w <- ww1) {
      if (ww2.contains(w)) {
        positiveScore += weights(w)
      }
      totalScore += weights(w)
    }

    for(w <- ww2) {
      if (!ww1.contains(w)) {
        totalScore += weights(w)
      }
    }

    positiveScore / totalScore
  }

  private def stem(word: String): String = {
    PotterStem(word)
  }

  private def tokenize(text: String): ArrayBuffer[String] = {
    val words = new ArrayBuffer[String]()

    var currentWord = ""
    for (c <- text) {
      if (c.isLetterOrDigit) {
        currentWord += c
      } else {
        if (c.isWhitespace) {
          if (currentWord != "") {
            words += currentWord
            currentWord = ""
          }
        }
      }
    }

    if (currentWord != "") {
      words += currentWord
    }

    words
  }
}
