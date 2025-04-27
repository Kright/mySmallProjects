
@main
def main(): Unit = {
  given ConsoleEffect:
    def print(msg: String): Unit = println(msg)

  printElems()
}

def printElems()(using console: ConsoleEffect): Unit =
  given EffectYield[Int] with
    def apply(v: Int): Boolean =
      console.print(s"processed $v")
      v <= 2

  given OnReturn[Int] with
    def apply(v: Int): Unit =
      console.print(s"stopped at $v")

  ReturnEffect.wrap {
    traverse(List(1, 2, 3, 4))
  }

  ReturnEffect.wrap {
    traverseV2(List(1, 2, 3, 4))
  }


trait EffectYield[T]:
  def apply(v: Int): Boolean

def traverse(lst: List[Int])(using effectYield: EffectYield[Int], returnEffect: ReturnEffect[Int]): Unit =
  lst match {
    case Nil => ()
    case x :: xs => if (effectYield(x)) traverse(xs) else returnEffect(x)
  }

def traverseV2(lst: List[Int])(using effectYield: EffectYield[Int], returnEffect: ReturnEffect[Int]): Unit = {
  lst.foreach { x =>
    if (!effectYield(x)) {
      returnEffect(x)
    }
  }
}

trait ConsoleEffect:
  def print(msg: String): Unit

trait ReturnEffect[T]:
  def apply(v: T): Nothing

trait OnReturn[T]:
  def apply(v: T): Unit

object ReturnEffect:
  def wrap[T](body: ReturnEffect[T] ?=> Unit)(using onReturn: OnReturn[T]): Unit = {
    case class ReturnException(result: T) extends RuntimeException()

    given ReturnEffect[T]:
      def apply(v: T): Nothing = throw ReturnException(v)

    try {
      body
    } catch {
      case ReturnException(result) => onReturn(result)
    }
  }
