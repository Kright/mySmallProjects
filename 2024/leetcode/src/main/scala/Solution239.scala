import scala.collection.mutable

object Solution239 extends App{
  println(maxSlidingWindow(Array(1, 3, -1, -3, 5, 3, 6, 7), 3).mkString(","))
  println(maxSlidingWindowV2(Array(1, 3, -1, -3, 5, 3, 6, 7), 3).mkString(","))

  trait BasketSortedTree {
    val start: Int
    def size: Int
    def elementsCount: Int
    def +=(v: Int): Unit
    def -=(v: Int): Unit
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

    def -=(v: Int): Unit = {
      array(v - start) -= 1
      elementsCount -= 1
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

    override def -=(v: Int): Unit = {
      val active = if (v >= right.start) right else left
      active -= v
      elementsCount -= 1
    }

    override def apply(index: Int): Int = {
      if (index < left.elementsCount) {
        left(index)
      } else {
        right(index - left.elementsCount)
      }
    }
  }

  def maxSlidingWindow(nums: Array[Int], k: Int): Array[Int] = {
    val result = new Array[Int](nums.length - k + 1)
    val tree = BasketSortedTree.make(-10000, 20001)

    for(i <- 0 until k) {
      tree += nums(i)
    }
    result(0) = tree(k - 1)

    for(i <- k until nums.length) {
      tree += nums(i)
      tree -= nums(i - k)
      result(i - k + 1) = tree(k - 1)
    }

    result
  }

  def maxSlidingWindowV2(nums: Array[Int], k: Int): Array[Int] = {
    val queue = new mutable.ArrayDeque[Int]()
    val result = new Array[Int](nums.length - k + 1)

    for(i <- 0 until k) {
      addToQueue(queue, nums(i))
    }

    result(0) = queue.head

    for(i <- k until nums.length) {
      addToQueue(queue, nums(i))

      if (nums(i - k) == queue.head) {
        queue.removeHead()
      }

      result(i - k + 1) = queue.head
    }

    result
  }

  private def addToQueue(queue: mutable.ArrayDeque[Int], n: Int): Unit = {
    while (queue.nonEmpty && queue.last < n) {
      queue.removeLast()
    }
    queue.addOne(n)
  }
}
