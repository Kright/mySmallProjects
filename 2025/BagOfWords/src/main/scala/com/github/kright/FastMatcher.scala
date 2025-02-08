package com.github.kright

final class FastMatcher private(private val orderedWords: Array[String], private val weights: Array[Double]) {

  def getSimilarity(other: FastMatcher): Double = {
    var pos1 = 0
    var pos2 = 0

    var positiveWeight: Double = 0.0
    var totalWeight: Double = 0.0

    while(true) {
      val isOk = pos1 < orderedWords.length && pos2 < other.orderedWords.length

      if (isOk) {
        val t1 = orderedWords(pos1)
        val t2 = other.orderedWords(pos2)

        if (t1 < t2) {
          totalWeight += weights(pos1)
          pos1 += 1
        } else if (t2 < t1) {
          totalWeight += other.weights(pos2)
          pos2 += 1
        } else {
          // t1 == t2
          val oldPos1 = pos1
          val oldPos2 = pos2
          pos1 = consume(t1, pos1 + 1)
          pos2 = other.consume(t2, pos2 + 1)
          val ww = weights(oldPos1)
          val count1 = pos1 - oldPos1
          val count2 = pos2 - oldPos2
          if (count1 <= count2) {
            positiveWeight += ww * count1
            totalWeight += ww * count2
          } else {
            positiveWeight += ww * count2
            totalWeight += ww * count1
          }
        }
      } else {
        totalWeight += getLastWeights(pos1)
        totalWeight += other.getLastWeights(pos2)
        return positiveWeight / totalWeight
      }
    }

    ???
  }

  private def consume(word: String, pos: Int): Int = {
    var i = pos
    while(i < orderedWords.length && orderedWords(i) == word) {
      i += 1
    }
    i
  }

  private def getLastWeights(pos: Int): Double = {
    var sum: Double = 0.0
    var i = pos
    while(i < weights.length) {
      sum += weights(i)
      i += 1
    }
    sum
  }
}

object FastMatcher {
  def apply(terms: Iterable[String], weightFunc: String => Double): FastMatcher = {
    val tt = terms.toSeq.sorted.toArray
    new FastMatcher(tt, tt.map(weightFunc))
  }
}
