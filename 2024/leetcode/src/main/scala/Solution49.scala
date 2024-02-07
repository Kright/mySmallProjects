object Solution49 extends App {

  println(groupAnagrams(Array("eat","tea","tan","ate","nat","bat")))

  def groupAnagrams(strs: Array[String]): List[List[String]] =
    strs.groupBy(_.sorted).toList.map{ case (_, arr) => arr.toList }
}
