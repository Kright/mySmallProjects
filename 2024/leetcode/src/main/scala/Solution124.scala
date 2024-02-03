object Solution124 extends App {
  class TreeNode(_value: Int = 0, _left: TreeNode = null, _right: TreeNode = null) {
    var value: Int = _value
    var left: TreeNode = _left
    var right: TreeNode = _right
  }


  def maxPathSum(root: TreeNode): Int =
    maxPathSums(root).getBest()

  private def max(a: Int, b: Int, c: Int): Int = math.max(a, math.max(b, c))

  case class Result(finished: Int, oneBranch: Int) {
    def getBest(): Int = math.max(finished, oneBranch)
  }

  private def maxPathSums(root: TreeNode): Result =
    if (root.left != null) {
      if (root.right != null) {
        maxPathSums(root.value, root.left, root.right)
      } else {
        maxPathSums(root.value, root.left)
      }
    } else {
      if (root.right != null) {
        maxPathSums(root.value, root.right)
      } else {
        maxPathSums(root.value)
      }
    }

  private def maxPathSums(value: Int, left: TreeNode, right: TreeNode): Result = {
    val l = maxPathSums(left)
    val r = maxPathSums(right)

    Result(
      finished = max(l.finished, r.finished, math.max(l.oneBranch, 0) + value + math.max(r.oneBranch, 0)),
      oneBranch = max(l.oneBranch + value, r.oneBranch + value, value)
    )
  }

  private def maxPathSums(value: Int, left: TreeNode): Result = {
    val r = maxPathSums(left)
    Result(
      finished = math.max(r.finished, value + math.max(r.oneBranch, 0)),
      oneBranch = math.max(value, r.oneBranch + value)
    )
  }

  private def maxPathSums(value: Int): Result = Result(value, value)
}
