import com.kright.MySequence

fun main() {
    val s = MySequence {
        var i = 0
        while (i < 5) {
            yield(i)
            i += 1
        }
    }

    println(s.toList())
    println(s.toList())

    val it = s.iterator()
    println(it.asSequence().toList())
    println(it.asSequence().toList())
}