# scalaLists

Benchmark of immutable single-linked lists vs "unrolled" (chunked) lists,
where each node stores up to 2, 4 or 8 elements in fields.
The idea: bigger nodes are more cache-friendly and faster.

Implementations:

* `scala.List` — baseline
* `MyList` — hand-written analog of `scala.List`, one element per node
* `List2` / `List4` / `List8` — up to 2 / 4 / 8 elements per node (`Node2` / `Node4` / `Node8`);
  only `e0` is always present, the rest of the slots are typed `A | Null`

All lists support:

* `::` — prepend (for a chunked list: copies the head node with `size + 1` elements,
  or starts a new node if the head node is full)
* `foreach` — iteration
* `length` — walks the nodes without touching the elements; chunked lists sum up node sizes
* `++` — concatenation (copies the left list iteratively, reuses the right one, like `scala.List`)
* pattern matching `case NodeN(head, tail) =>` — a name-based `unapply` without Option/Tuple
  allocation, which returns the head and dynamically creates a tail node with the elements
  shifted by one (or returns `next` if the head node had only one element)

Elements are `Box(value: Int)` objects allocated sequentially. With the `shuffled=true`
parameter the array of boxes is shuffled (fixed seed) before the lists are built, so
iteration has to jump around in memory for every element. The point: a chunked node holds
up to 8 independent references, so the CPU can prefetch several elements in parallel,
while a cons list is one long serial dependency chain.

`Array[Box]` is included as the speed-of-light baseline (`sumWhile_array`), plus
`length_array` (O(1)) and build/concat via `arraycopy`.

## How to run

```shell
sbt test
```

```shell
sbt "Jmh/run"
```

Only a subset, single fork, one parameter combination:

```shell
sbt "Jmh/run -f 1 -p size=1000000 -p shuffled=true sumForeach.*"
```

In case of stale-class errors after renaming things: `sbt clean`.

## Results

AMD Ryzen 9 5950X, OpenJDK 25.0.3, Ubuntu 24.04, single thread.

```
Benchmark                           (shuffled)   (size)  Mode  Cnt        Score         Error  Units
ListBenchmark.build_array                false     1000  avgt   10     1007.032 ±       6.034  ns/op
ListBenchmark.build_array                false  1000000  avgt   10  1000064.222 ±    7111.057  ns/op
ListBenchmark.build_array                 true     1000  avgt   10     1028.220 ±      10.566  ns/op
ListBenchmark.build_array                 true  1000000  avgt   10  1013024.557 ±    8509.199  ns/op
ListBenchmark.build_list2                false     1000  avgt   10     2507.363 ±      51.839  ns/op
ListBenchmark.build_list2                false  1000000  avgt   10  2728780.843 ±  639223.572  ns/op
ListBenchmark.build_list2                 true     1000  avgt   10     2496.114 ±      80.097  ns/op
ListBenchmark.build_list2                 true  1000000  avgt   10  2940249.522 ±  852640.757  ns/op
ListBenchmark.build_list4                false     1000  avgt   10     3250.588 ±     105.314  ns/op
ListBenchmark.build_list4                false  1000000  avgt   10  3574355.524 ±  678293.729  ns/op
ListBenchmark.build_list4                 true     1000  avgt   10     3221.087 ±      31.906  ns/op
ListBenchmark.build_list4                 true  1000000  avgt   10  3496821.086 ±  808467.165  ns/op
ListBenchmark.build_list8                false     1000  avgt   10     4261.440 ±     307.762  ns/op
ListBenchmark.build_list8                false  1000000  avgt   10  4566756.799 ±  199514.442  ns/op
ListBenchmark.build_list8                 true     1000  avgt   10     4377.056 ±      99.229  ns/op
ListBenchmark.build_list8                 true  1000000  avgt   10  4679005.005 ±  830759.592  ns/op
ListBenchmark.build_myList               false     1000  avgt   10     1899.800 ±      17.747  ns/op
ListBenchmark.build_myList               false  1000000  avgt   10  1874049.879 ±  435638.929  ns/op
ListBenchmark.build_myList                true     1000  avgt   10     1895.743 ±      20.663  ns/op
ListBenchmark.build_myList                true  1000000  avgt   10  1797290.407 ±  307953.497  ns/op
ListBenchmark.build_scalaList            false     1000  avgt   10     1719.687 ±      27.923  ns/op
ListBenchmark.build_scalaList            false  1000000  avgt   10  1921691.979 ±  682383.379  ns/op
ListBenchmark.build_scalaList             true     1000  avgt   10     1713.155 ±      22.616  ns/op
ListBenchmark.build_scalaList             true  1000000  avgt   10  1822329.177 ±   85417.081  ns/op
ListBenchmark.concat_array               false     1000  avgt   10      632.093 ±      11.337  ns/op
ListBenchmark.concat_array               false  1000000  avgt   10  1028762.986 ±   48356.709  ns/op
ListBenchmark.concat_array                true     1000  avgt   10      630.866 ±       9.591  ns/op
ListBenchmark.concat_array                true  1000000  avgt   10  1030900.114 ±   37484.761  ns/op
ListBenchmark.concat_list2               false     1000  avgt   10     1283.170 ±      25.591  ns/op
ListBenchmark.concat_list2               false  1000000  avgt   10  1857190.776 ±  523208.605  ns/op
ListBenchmark.concat_list2                true     1000  avgt   10     1290.084 ±      46.099  ns/op
ListBenchmark.concat_list2                true  1000000  avgt   10  2386776.041 ±  890015.417  ns/op
ListBenchmark.concat_list4               false     1000  avgt   10      810.954 ±      30.028  ns/op
ListBenchmark.concat_list4               false  1000000  avgt   10   921732.232 ±  193646.249  ns/op
ListBenchmark.concat_list4                true     1000  avgt   10      811.487 ±      68.071  ns/op
ListBenchmark.concat_list4                true  1000000  avgt   10   845451.039 ±  161941.829  ns/op
ListBenchmark.concat_list8               false     1000  avgt   10      561.381 ±      28.050  ns/op
ListBenchmark.concat_list8               false  1000000  avgt   10   568822.656 ±   13134.710  ns/op
ListBenchmark.concat_list8                true     1000  avgt   10      550.418 ±      16.492  ns/op
ListBenchmark.concat_list8                true  1000000  avgt   10   601987.487 ±  130376.643  ns/op
ListBenchmark.concat_myList              false     1000  avgt   10     2132.696 ±      73.838  ns/op
ListBenchmark.concat_myList              false  1000000  avgt   10  3657559.580 ± 1464869.124  ns/op
ListBenchmark.concat_myList               true     1000  avgt   10     2125.092 ±      97.673  ns/op
ListBenchmark.concat_myList               true  1000000  avgt   10  6354883.185 ± 1117932.627  ns/op
ListBenchmark.concat_scalaList           false     1000  avgt   10     3398.908 ±      68.005  ns/op
ListBenchmark.concat_scalaList           false  1000000  avgt   10  4039392.988 ±  874291.782  ns/op
ListBenchmark.concat_scalaList            true     1000  avgt   10     3286.550 ±      25.262  ns/op
ListBenchmark.concat_scalaList            true  1000000  avgt   10  6369585.390 ± 1611323.653  ns/op
ListBenchmark.length_array               false     1000  avgt   10        0.449 ±       0.017  ns/op
ListBenchmark.length_array               false  1000000  avgt   10        0.451 ±       0.005  ns/op
ListBenchmark.length_array                true     1000  avgt   10        0.447 ±       0.004  ns/op
ListBenchmark.length_array                true  1000000  avgt   10        0.451 ±       0.007  ns/op
ListBenchmark.length_list2               false     1000  avgt   10      652.121 ±       6.448  ns/op
ListBenchmark.length_list2               false  1000000  avgt   10   855119.781 ±   91208.593  ns/op
ListBenchmark.length_list2                true     1000  avgt   10      653.947 ±      14.210  ns/op
ListBenchmark.length_list2                true  1000000  avgt   10   890589.523 ±   25060.949  ns/op
ListBenchmark.length_list4               false     1000  avgt   10      328.076 ±       2.722  ns/op
ListBenchmark.length_list4               false  1000000  avgt   10   331066.196 ±    7368.275  ns/op
ListBenchmark.length_list4                true     1000  avgt   10      327.255 ±       3.168  ns/op
ListBenchmark.length_list4                true  1000000  avgt   10   330811.333 ±    3040.209  ns/op
ListBenchmark.length_list8               false     1000  avgt   10      168.926 ±       3.518  ns/op
ListBenchmark.length_list8               false  1000000  avgt   10   165240.600 ±     523.091  ns/op
ListBenchmark.length_list8                true     1000  avgt   10      166.536 ±       1.467  ns/op
ListBenchmark.length_list8                true  1000000  avgt   10   165077.688 ±     466.856  ns/op
ListBenchmark.length_myList              false     1000  avgt   10     1210.387 ±      24.112  ns/op
ListBenchmark.length_myList              false  1000000  avgt   10  1770477.106 ±  262370.730  ns/op
ListBenchmark.length_myList               true     1000  avgt   10     1202.917 ±      17.818  ns/op
ListBenchmark.length_myList               true  1000000  avgt   10  3957572.050 ±  407199.742  ns/op
ListBenchmark.length_scalaList           false     1000  avgt   10     1126.348 ±       7.926  ns/op
ListBenchmark.length_scalaList           false  1000000  avgt   10  1709198.402 ±  191268.835  ns/op
ListBenchmark.length_scalaList            true     1000  avgt   10     1127.030 ±       7.216  ns/op
ListBenchmark.length_scalaList            true  1000000  avgt   10  3339247.789 ±  849782.644  ns/op
ListBenchmark.sumForeach_array           false     1000  avgt   10      237.988 ±       3.134  ns/op
ListBenchmark.sumForeach_array           false  1000000  avgt   10   771686.449 ±  105801.537  ns/op
ListBenchmark.sumForeach_array            true     1000  avgt   10      235.726 ±       2.561  ns/op
ListBenchmark.sumForeach_array            true  1000000  avgt   10   704890.865 ±  207493.171  ns/op
ListBenchmark.sumForeach_list2           false     1000  avgt   10      798.592 ±       5.454  ns/op
ListBenchmark.sumForeach_list2           false  1000000  avgt   10  2390412.421 ±  184563.451  ns/op
ListBenchmark.sumForeach_list2            true     1000  avgt   10      796.265 ±       2.733  ns/op
ListBenchmark.sumForeach_list2            true  1000000  avgt   10  3061780.377 ±  385863.058  ns/op
ListBenchmark.sumForeach_list4           false     1000  avgt   10      512.510 ±       4.337  ns/op
ListBenchmark.sumForeach_list4           false  1000000  avgt   10  1390459.152 ±  210061.672  ns/op
ListBenchmark.sumForeach_list4            true     1000  avgt   10      507.536 ±       2.670  ns/op
ListBenchmark.sumForeach_list4            true  1000000  avgt   10  1262051.452 ±   62334.740  ns/op
ListBenchmark.sumForeach_list8           false     1000  avgt   10      434.767 ±       2.260  ns/op
ListBenchmark.sumForeach_list8           false  1000000  avgt   10   962749.109 ±   48897.117  ns/op
ListBenchmark.sumForeach_list8            true     1000  avgt   10      432.729 ±       1.482  ns/op
ListBenchmark.sumForeach_list8            true  1000000  avgt   10   936717.248 ±   34181.192  ns/op
ListBenchmark.sumForeach_myList          false     1000  avgt   10     1404.306 ±       5.920  ns/op
ListBenchmark.sumForeach_myList          false  1000000  avgt   10  3409104.863 ±   69988.686  ns/op
ListBenchmark.sumForeach_myList           true     1000  avgt   10     1399.796 ±       4.442  ns/op
ListBenchmark.sumForeach_myList           true  1000000  avgt   10  4639849.413 ±  393904.078  ns/op
ListBenchmark.sumForeach_scalaList       false     1000  avgt   10     1294.026 ±       9.163  ns/op
ListBenchmark.sumForeach_scalaList       false  1000000  avgt   10  3342175.690 ±   71464.519  ns/op
ListBenchmark.sumForeach_scalaList        true     1000  avgt   10     1278.054 ±       2.539  ns/op
ListBenchmark.sumForeach_scalaList        true  1000000  avgt   10  4657472.002 ±  464723.468  ns/op
ListBenchmark.sumUnapply_list2           false     1000  avgt   10     1523.016 ±      16.272  ns/op
ListBenchmark.sumUnapply_list2           false  1000000  avgt   10  3198774.700 ±   64639.416  ns/op
ListBenchmark.sumUnapply_list2            true     1000  avgt   10     1448.510 ±      73.382  ns/op
ListBenchmark.sumUnapply_list2            true  1000000  avgt   10  3669768.583 ±  212820.091  ns/op
ListBenchmark.sumUnapply_list4           false     1000  avgt   10     2301.507 ±      16.856  ns/op
ListBenchmark.sumUnapply_list4           false  1000000  avgt   10  3496507.044 ±   55901.554  ns/op
ListBenchmark.sumUnapply_list4            true     1000  avgt   10     2289.181 ±      13.075  ns/op
ListBenchmark.sumUnapply_list4            true  1000000  avgt   10  3452323.811 ±   72695.532  ns/op
ListBenchmark.sumUnapply_list8           false     1000  avgt   10     4098.592 ±     100.549  ns/op
ListBenchmark.sumUnapply_list8           false  1000000  avgt   10  6642944.761 ±  415947.970  ns/op
ListBenchmark.sumUnapply_list8            true     1000  avgt   10     4195.838 ±      96.449  ns/op
ListBenchmark.sumUnapply_list8            true  1000000  avgt   10  6722229.831 ±  576939.918  ns/op
ListBenchmark.sumUnapply_myList          false     1000  avgt   10     1363.610 ±       8.494  ns/op
ListBenchmark.sumUnapply_myList          false  1000000  avgt   10  3643070.620 ±   49776.368  ns/op
ListBenchmark.sumUnapply_myList           true     1000  avgt   10     1364.829 ±       9.546  ns/op
ListBenchmark.sumUnapply_myList           true  1000000  avgt   10  5138059.050 ±  241033.157  ns/op
ListBenchmark.sumUnapply_scalaList       false     1000  avgt   10     1197.532 ±       6.915  ns/op
ListBenchmark.sumUnapply_scalaList       false  1000000  avgt   10  3538030.814 ±   48755.237  ns/op
ListBenchmark.sumUnapply_scalaList        true     1000  avgt   10     1183.180 ±      11.382  ns/op
ListBenchmark.sumUnapply_scalaList        true  1000000  avgt   10  5264814.307 ±  139411.019  ns/op
ListBenchmark.sumWhile_array             false     1000  avgt   10      214.546 ±       3.359  ns/op
ListBenchmark.sumWhile_array             false  1000000  avgt   10   741468.263 ±   90859.302  ns/op
ListBenchmark.sumWhile_array              true     1000  avgt   10      214.183 ±       2.389  ns/op
ListBenchmark.sumWhile_array              true  1000000  avgt   10   791676.223 ±  149828.828  ns/op
```

In short:

* iteration (`sumForeach`, 1M elements): `list8` is 3.5x faster than `scala.List` on the
  sequential layout (0.94 ms vs 3.3 ms) and 5x faster on the shuffled one (0.94 ms vs 4.7 ms),
  staying close to `Array[Box]` (~0.7-0.8 ms).
* the shuffle barely affects `list4` / `list8` at all, while cons lists lose ~40%:
  up to 8 independent references per node give the CPU enough memory-level parallelism
  to hide the latency of random element accesses; a cons list is a serial pointer chain.
* `length`: chunked lists only sum up per-node sizes without touching elements —
  `list8` is 10-20x faster than `scala.List` (0.165 ms vs 1.7-3.3 ms at 1M);
  `Array` is O(1) and out of reach.
* head/tail decomposition via `unapply` is where chunked lists pay: every step allocates
  a tail node and copies the whole chunk, so the cost grows with the chunk size
  (`list8` 4.1 µs vs 1.2 µs for `scala.List` at size=1000).
* building by prepending one element at a time: chunked lists are 1.3-2.5x slower than
  `scala.List` (each prepend copies the head node, most allocations become garbage).
* concatenation: `list8` is ~6x faster than `scala.List` (fewer nodes to copy),
  close to `arraycopy`.
* a curious GC side effect: with `shuffled=true` even benchmarks that never touch the
  elements (`length_myList`, `concat_myList`) get ~2x slower at 1M for the cons lists —
  during evacuation the copying GC relocates list nodes interleaved with the shuffled
  boxes they reference, scattering the node chain itself. Chunked lists suffer much less,
  having 2-8x fewer nodes.
* `build` / `concat` numbers at 1M have large error bars — that is GC pressure.
