package com.github.kright

import scala.collection.mutable

trait SearchIndex[K, V] {
  def apply(key: K): Set[V]

  def apply(keys: IterableOnce[K]): Set[V] = {
    var result = Set[V]()
    for(key <- keys.iterator) {
      result = result.union(this(key))
    }
    result
  }
}

object SearchIndex {
  def apply[K, V](data: IterableOnce[(IterableOnce[K], V)]): SearchIndex[K, V] = {
    val index = new mutable.HashMap[K, mutable.HashSet[V]]()

    for((keys, v) <- data.iterator) {
      for(key <- keys.iterator) {
        val set = index.getOrElseUpdate(key, new mutable.HashSet[V]())
        set.add(v)
      }
    }

    new SearchIndex[K, V] {
      override def apply(key: K): Set[V] = index.getOrElse(key, Set()).toSet

      override def apply(keys: IterableOnce[K]): Set[V] = {
        val result = new mutable.HashSet[V]()
        for(key <- keys.iterator) {
          index.get(key).foreach { values =>
            result.addAll(values)
          }
        }
        result.toSet
      }
    }
  }
}
