/**
 * Definition for singly-linked list.
 * class ListNode(_x: Int = 0, _next: ListNode = null) {
 *   var next: ListNode = _next
 *   var x: Int = _x
 * }
 */
class ListNode(_x: Int = 0, _next: ListNode = null) {
  var next: ListNode = _next
  var x: Int = _x

  override def toString: String = s"$x :: $next"
}

object Solution23 extends App {

  println(mergeKLists(
    Array(
      new ListNode(1, new ListNode(4, new ListNode(5))),
      new ListNode(1, new ListNode(3, new ListNode(4))),
      new ListNode(2, new ListNode(6)),
      null,
    )
  ))

  def mergeKLists(lists: Array[ListNode]): ListNode = {
    implicit val listOrdering: Ordering[ListNode] = (left: ListNode, right: ListNode) => right.x.compare(left.x)

    val heap = collection.mutable.PriorityQueue.empty[ListNode]

    for(list <- lists) {
      if (list != null) {
        heap.addOne(list)
      }
    }

    val result = new ListNode(0, null)
    var resultTail = result

    while(heap.nonEmpty) {
      val current = heap.dequeue()
      if (current.next != null) {
        heap.addOne(current.next)
      } else {
        current.next = null
      }
      resultTail.next = current
      resultTail = current
    }

    result.next
  }
}
