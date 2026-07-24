package me.kright.lists

import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.util.concurrent.TimeUnit
import scala.compiletime.uninitialized

/** list element; a separate object so that the element layout in memory can be shuffled */
final class Box(val value: Int)

@State(Scope.Thread)
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(2)
class ListBenchmark {

  @Param(Array("1000", "1000000"))
  var size: Int = uninitialized

  /**
   * false: boxes are laid out in memory in the list order, iteration walks memory sequentially.
   * true: the array of boxes is shuffled before building the lists, so iteration jumps
   * around in memory. The chunked lists keep up to 8 independent references per node,
   * so the CPU has more freedom to prefetch elements in parallel; in a cons list
   * every node adds a serial dependency to the chain.
   */
  @Param(Array("false", "true"))
  var shuffled: Boolean = uninitialized

  var boxes: Array[Box] = uninitialized
  var scalaList: List[Box] = uninitialized
  var myList: MyList[Box] = uninitialized
  var list2: List2[Box] = uninitialized
  var list4: List4[Box] = uninitialized
  var list8: List8[Box] = uninitialized

  // each list is built in its own loop so that its nodes are allocated
  // compactly, without interleaving with nodes of the other lists
  @Setup
  def setup(): Unit = {
    boxes = Array.tabulate(size)(new Box(_))

    if (shuffled) {
      val random = new java.util.Random(42)
      var i = size - 1
      while (i > 0) {
        val j = random.nextInt(i + 1)
        val t = boxes(i)
        boxes(i) = boxes(j)
        boxes(j) = t
        i -= 1
      }
    }

    var sc: List[Box] = Nil
    var i = size - 1
    while (i >= 0) {
      sc = boxes(i) :: sc
      i -= 1
    }
    scalaList = sc

    var my: MyList[Box] = MyNil
    i = size - 1
    while (i >= 0) {
      my = boxes(i) :: my
      i -= 1
    }
    myList = my

    var l2: List2[Box] = Nil2
    i = size - 1
    while (i >= 0) {
      l2 = boxes(i) :: l2
      i -= 1
    }
    list2 = l2

    var l4: List4[Box] = Nil4
    i = size - 1
    while (i >= 0) {
      l4 = boxes(i) :: l4
      i -= 1
    }
    list4 = l4

    var l8: List8[Box] = Nil8
    i = size - 1
    while (i >= 0) {
      l8 = boxes(i) :: l8
      i -= 1
    }
    list8 = l8

    // compact the surviving nodes, removing the garbage created during building;
    // note: a copying GC may partially rearrange the boxes, so the shuffled
    // layout is best-effort
    System.gc()
  }

  // build: prepend `size` elements to an empty list

  @Benchmark
  def build_array(bh: Blackhole): Unit = {
    val src = boxes
    val a = new Array[Box](src.length)
    var i = 0
    while (i < src.length) {
      a(i) = src(i)
      i += 1
    }
    bh.consume(a)
  }

  @Benchmark
  def build_scalaList(bh: Blackhole): Unit = {
    val src = boxes
    var l: List[Box] = Nil
    var i = src.length - 1
    while (i >= 0) {
      l = src(i) :: l
      i -= 1
    }
    bh.consume(l)
  }

  @Benchmark
  def build_myList(bh: Blackhole): Unit = {
    val src = boxes
    var l: MyList[Box] = MyNil
    var i = src.length - 1
    while (i >= 0) {
      l = src(i) :: l
      i -= 1
    }
    bh.consume(l)
  }

  @Benchmark
  def build_list2(bh: Blackhole): Unit = {
    val src = boxes
    var l: List2[Box] = Nil2
    var i = src.length - 1
    while (i >= 0) {
      l = src(i) :: l
      i -= 1
    }
    bh.consume(l)
  }

  @Benchmark
  def build_list4(bh: Blackhole): Unit = {
    val src = boxes
    var l: List4[Box] = Nil4
    var i = src.length - 1
    while (i >= 0) {
      l = src(i) :: l
      i -= 1
    }
    bh.consume(l)
  }

  @Benchmark
  def build_list8(bh: Blackhole): Unit = {
    val src = boxes
    var l: List8[Box] = Nil8
    var i = src.length - 1
    while (i >= 0) {
      l = src(i) :: l
      i -= 1
    }
    bh.consume(l)
  }

  // sum of elements via foreach; sumWhile_array is the speed-of-light baseline

  @Benchmark
  def sumWhile_array(bh: Blackhole): Unit = {
    val a = boxes
    var sum = 0
    var i = 0
    while (i < a.length) {
      sum += a(i).value
      i += 1
    }
    bh.consume(sum)
  }

  @Benchmark
  def sumForeach_array(bh: Blackhole): Unit = {
    var sum = 0
    boxes.foreach(sum += _.value)
    bh.consume(sum)
  }

  @Benchmark
  def sumForeach_scalaList(bh: Blackhole): Unit = {
    var sum = 0
    scalaList.foreach(sum += _.value)
    bh.consume(sum)
  }

  @Benchmark
  def sumForeach_myList(bh: Blackhole): Unit = {
    var sum = 0
    myList.foreach(sum += _.value)
    bh.consume(sum)
  }

  @Benchmark
  def sumForeach_list2(bh: Blackhole): Unit = {
    var sum = 0
    list2.foreach(sum += _.value)
    bh.consume(sum)
  }

  @Benchmark
  def sumForeach_list4(bh: Blackhole): Unit = {
    var sum = 0
    list4.foreach(sum += _.value)
    bh.consume(sum)
  }

  @Benchmark
  def sumForeach_list8(bh: Blackhole): Unit = {
    var sum = 0
    list8.foreach(sum += _.value)
    bh.consume(sum)
  }

  // sum of elements via pattern matching (unapply, biting off the head);
  // for chunked lists this dynamically allocates a tail node per step

  @Benchmark
  def sumUnapply_scalaList(bh: Blackhole): Unit = {
    var sum = 0
    var current = scalaList
    var continue = true
    while (continue) {
      current match {
        case h :: t =>
          sum += h.value
          current = t
        case _ =>
          continue = false
      }
    }
    bh.consume(sum)
  }

  @Benchmark
  def sumUnapply_myList(bh: Blackhole): Unit = {
    var sum = 0
    var current = myList
    var continue = true
    while (continue) {
      current match {
        case MyCons(h, t) =>
          sum += h.value
          current = t
        case _ =>
          continue = false
      }
    }
    bh.consume(sum)
  }

  @Benchmark
  def sumUnapply_list2(bh: Blackhole): Unit = {
    var sum = 0
    var current = list2
    var continue = true
    while (continue) {
      current match {
        case Node2(h, t) =>
          sum += h.value
          current = t
        case _ =>
          continue = false
      }
    }
    bh.consume(sum)
  }

  @Benchmark
  def sumUnapply_list4(bh: Blackhole): Unit = {
    var sum = 0
    var current = list4
    var continue = true
    while (continue) {
      current match {
        case Node4(h, t) =>
          sum += h.value
          current = t
        case _ =>
          continue = false
      }
    }
    bh.consume(sum)
  }

  @Benchmark
  def sumUnapply_list8(bh: Blackhole): Unit = {
    var sum = 0
    var current = list8
    var continue = true
    while (continue) {
      current match {
        case Node8(h, t) =>
          sum += h.value
          current = t
        case _ =>
          continue = false
      }
    }
    bh.consume(sum)
  }

  // length: walks the nodes without touching the elements;
  // chunked lists just sum up the node sizes, ~2/4/8x fewer steps

  @Benchmark
  def length_array(bh: Blackhole): Unit =
    bh.consume(boxes.length)

  @Benchmark
  def length_scalaList(bh: Blackhole): Unit =
    bh.consume(scalaList.length)

  @Benchmark
  def length_myList(bh: Blackhole): Unit =
    bh.consume(myList.length)

  @Benchmark
  def length_list2(bh: Blackhole): Unit =
    bh.consume(list2.length)

  @Benchmark
  def length_list4(bh: Blackhole): Unit =
    bh.consume(list4.length)

  @Benchmark
  def length_list8(bh: Blackhole): Unit =
    bh.consume(list8.length)

  // concatenation: copies the left list, reuses the right one
  // (element layout should not matter here, elements are never dereferenced)

  @Benchmark
  def concat_array(bh: Blackhole): Unit = {
    val a = boxes
    val result = new Array[Box](a.length * 2)
    System.arraycopy(a, 0, result, 0, a.length)
    System.arraycopy(a, 0, result, a.length, a.length)
    bh.consume(result)
  }

  @Benchmark
  def concat_scalaList(bh: Blackhole): Unit =
    bh.consume(scalaList ::: scalaList)

  @Benchmark
  def concat_myList(bh: Blackhole): Unit =
    bh.consume(myList ++ myList)

  @Benchmark
  def concat_list2(bh: Blackhole): Unit =
    bh.consume(list2 ++ list2)

  @Benchmark
  def concat_list4(bh: Blackhole): Unit =
    bh.consume(list4 ++ list4)

  @Benchmark
  def concat_list8(bh: Blackhole): Unit =
    bh.consume(list8 ++ list8)
}
