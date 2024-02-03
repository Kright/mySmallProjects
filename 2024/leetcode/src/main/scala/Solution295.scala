import scala.collection.mutable

object Solution295 extends App {
  val medianFinder = new MedianFinder
  medianFinder.addNum(1) // arr = [1]
  println(medianFinder.findMedian) // return 1

  medianFinder.addNum(2) // arr = [1, 2]
  println(medianFinder.findMedian) // return 1.5 (i.e., (1 + 2) / 2)
  medianFinder.addNum(3) // arr[1, 2, 3]
  println(medianFinder.findMedian) // return 2.0

  trait BasketSortedTree {
    val start: Int
    def size: Int
    def elementsCount: Int
    def +=(v: Int): Unit
    def apply(index: Int): Int
  }

  object BasketSortedTree {
    def make(start: Int, size: Int): BasketSortedTree =
      if (size > 32) {
        new BasketTreeNode(start, size)
      } else {
        new Basket(start, size)
      }
  }

  class Basket(override val start: Int, private val array: Array[Int]) extends BasketSortedTree {
    def this(start: Int, size: Int) = this(start, new Array[Int](size))

    override def size: Int = array.length

    var elementsCount: Int = array.sum

    def +=(v: Int): Unit = {
      array(v - start) += 1
      elementsCount += 1
    }

    def apply(index: Int): Int = {
      require(index < elementsCount)
      var count = 0
      for (j <- array.indices) {
        count += array(j)
        if (count > index) {
          return j + start
        }
      }
      ???
    }
  }

  class BasketTreeNode(override val start: Int,
                       override val size: Int) extends BasketSortedTree {

    private lazy val left: BasketSortedTree = BasketSortedTree.make(start, size / 2)
    private lazy val right: BasketSortedTree = BasketSortedTree.make(start + size / 2, size - size / 2)

    var elementsCount = 0

    override def +=(v: Int): Unit = {
      val active = if (v >= right.start) right else left
      active += v
      elementsCount += 1
    }

    override def apply(index: Int): Int = {
      if (index < left.elementsCount) {
        left(index)
      } else {
        right(index - left.elementsCount)
      }
    }
  }

  class MedianFinder() {
    private val data = BasketSortedTree.make(-100000, 200001)

    def addNum(num: Int): Unit = {
      data += num
    }

    def findMedian(): Double = {
      val total = data.elementsCount
      if (total % 2 == 1) {
        data(total / 2)
      } else {
        0.5 * (data(total / 2) + data(total / 2 - 1))
      }
    }
  }

  class MedianFinderV2() {
    private val high = scala.collection.mutable.PriorityQueue[Int]()
    private val low = scala.collection.mutable.PriorityQueue[Int]().reverse

    def addNum(num: Int): Unit = {
      low.addOne(num)
      high.addOne(low.dequeue())
      if (low.length + 1 < high.length) low.addOne(high.dequeue())
    }

    def findMedian(): Double = {
      if (high.length != low.length) {
        high.head
      } else {
        0.5 * (low.head + high.head)
      }
    }
  }
}

