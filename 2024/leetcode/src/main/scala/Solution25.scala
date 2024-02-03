import scala.annotation.tailrec

object Solution25 extends App{
  println(reverseKGroup(new ListNode(0, new ListNode(1, new ListNode(2, new ListNode(3)))), k = 2))
  println(reverseKGroup(new ListNode(0, new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4))))), k = 2))

  class ListNode(_x: Int = 0, _next: ListNode = null) {
    var next: ListNode = _next
    var x: Int = _x

    override def toString: String = s"$x :: $next"
  }

  def reverseKGroup(head: ListNode, k: Int): ListNode = {
    val tail = detachTail(head, k)

    if (tail == null && getLen(head) < k) {
      return head
    }
    val reversedHead = reverse(head)
    val last = head
    last.next = reverseKGroup(tail, k)
    reversedHead
  }

  @tailrec
  def detachTail(head: ListNode, k: Int): ListNode = {
    require(k > 0)
    if (head == null) return null
    if (k == 1) {
      val tail = head.next
      head.next = null
      return tail
    }
    if (head.next == null) return null
    detachTail(head.next, k - 1)
  }

  def getLen(head: ListNode): Int =
    if (head == null) 0
    else getLen(head.next) + 1

  def reverse(head: ListNode): ListNode = {
    var current: ListNode = null
    var tail: ListNode = head

    while (tail != null) {
      val oldTail = tail
      tail = tail.next

      oldTail.next = current
      current = oldTail
    }

    current
  }
}
