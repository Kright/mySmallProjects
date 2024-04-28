import scala.collection.mutable

object Solution1488 extends App {
  println(avoidFlood(Array(1, 2, 0, 0, 0, 2, 1)).mkString(", "))

  def avoidFlood(rains: Array[Int]): Array[Int] = {
    val fullLakes = new mutable.HashSet[Int]()

    val nextRain = new Array[Int](rains.length)
    val nextRainPos = new mutable.HashMap[Int, Int]()
    for (i <- rains.indices.reverse) {
      val lake = rains(i)
      if (lake > 0) {
        if (nextRainPos.contains(lake)) {
          nextRain(i) = nextRainPos(lake)
        }
        nextRainPos(lake) = i
      }
    }

    val orders = mutable.PriorityQueue.empty[ClearOrder]
    val result = new Array[Int](rains.length)

    val anyLake = rains.find(_ > 0).get

    for ((rain, i) <- rains.zipWithIndex) {
      if (rain > 0) {
        val lake = rain
        if (fullLakes.contains(lake)) {
          return Array()
        }
        fullLakes.add(lake)
        if (nextRain(i) != 0) {
          orders += ClearOrder(lake, nextRain(i))
        }
        result(i) = -1
      } else { // rain == 0
        val nextLake = if (orders.nonEmpty) orders.dequeue().lake else anyLake
        fullLakes.remove(nextLake)
        result(i) = nextLake
      }
    }

    result
  }

  private case class ClearOrder(lake: Int, lastMoment: Int) extends Ordered[ClearOrder] {
    override def compare(other: ClearOrder): Int = other.lastMoment - this.lastMoment
  }
}
