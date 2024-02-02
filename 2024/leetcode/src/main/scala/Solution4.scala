import scala.collection.mutable.ArrayBuffer

class Solution4 extends App {

  def findMedianSortedArrays(nums1: Array[Int], nums2: Array[Int]): Double = {
    var low1 = 0
    var low2 = 0
    var up1 = nums1.length - 1
    var up2 = nums2.length - 1
    val totalElements = nums1.length + nums2.length

    for (_ <- 0 until ((totalElements - 1) / 2)) {
      if (low1 <= up1) {
        if (low2 <= up2) {
          if (nums1(low1) < nums2(low2)) low1 += 1
          else low2 += 1
        } else {
          low1 += 1
        }
      } else {
        low2 += 1
      }

      if (low1 <= up1) {
        if (low2 <= up2) {
          if (nums1(up1) < nums2(up2)) up2 -= 1
          else up1 -= 1
        } else {
          up1 -= 1
        }
      } else {
        up2 -= 1
      }
    }

    if (totalElements % 2 == 1) {
      if (low1 <= up1) nums1(low1) else nums2(low2)
    } else {
      var elements: Double = 0.0

      for (i <- low1 to up1) {
        elements += nums1(i)
      }
      for (i <- low2 to up2) {
        elements += nums2(i)
      }

      elements / 2.0
    }
  }
}
