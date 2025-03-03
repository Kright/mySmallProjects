package test

class Matrix4x4:
  inline val w = 4
  inline val h = 4
  val elements: Array[Double] = new Array[Double](16)

  def fillRandom(): Unit = {
    for (i <- elements.indices) {
      elements(i) = Math.random()
    }
  }

  def apply(y: Int, x: Int): Double =
    elements(y * w + x)

  def update(y: Int, x: Int, v: Double): Unit =
    elements(y * w + x) = v

  def multiplyNaive(m: Matrix4x4): Matrix4x4 = {
    val result = new Matrix4x4()

    for (y <- 0 until h) {
      for (x <- 0 until w) {
        var sum = 0.0
        for (k <- 0 until w) {
          sum += this (y, k) * m(k, x)
        }
        result(y, x) = sum
      }
    }

    result
  }

  def multiplyCfor(m: Matrix4x4): Matrix4x4 = {
    val result = new Matrix4x4()

    FastRange.cfor(0, _ < h, _ + 1) { y =>
      FastRange.cfor(0, _ < w, _ + 1) { x =>
        var sum = 0.0
        FastRange.cfor(0, _ < w, _ + 1) { k =>
          sum += this (y, k) * m(k, x)
        }
        result(y, x) = sum
      }
    }

    result
  }

  def multiplyCustomLoop(m: Matrix4x4): Matrix4x4 = {
    val result = new Matrix4x4()

    loop(4) { y =>
      loop(4) { x =>
        var sum = 0.0
        loop(4) { k =>
          sum += this (y, k) * m(k, x)
        }
        result(y, x) = sum
      }
    }

    result
  }

  def multiplyNaiveUnrolled(m: Matrix4x4): Matrix4x4 = {
    val result = new Matrix4x4()

    for (y <- 0 until h) {
      for (x <- 0 until w) {
        result(y, x) =
          this (y, 0) * m(0, x) +
            this (y, 1) * m(1, x) +
            this (y, 2) * m(2, x) +
            this (y, 3) * m(3, x)
      }
    }

    result
  }

  def multiplyFma(m: Matrix4x4): Matrix4x4 = {
    val result = new Matrix4x4()

    for (y <- 0 until h) {
      for (x <- 0 until w) {
        var sum = 0.0
        for (k <- 0 until w) {
          sum = Math.fma(this (y, k), m(k, x), sum)
        }
        result(y, x) = sum
      }
    }

    result
  }

  def multiplyFmaCustomLoop(m: Matrix4x4): Matrix4x4 = {
    val result = new Matrix4x4()

    loop(4) { y =>
      loop(4) { x =>
        var sum = 0.0
        loop(4) { k =>
          sum = Math.fma(this (y, k), m(k, x), sum)
        }
        result(y, x) = sum
      }
    }

    result
  }

  def multiplyFmaFastRange(m: Matrix4x4): Matrix4x4 = {
    val result = new Matrix4x4()

    for (y <- FastRange(4)) {
      for (x <- FastRange(4)) {
        var sum = 0.0
        for (k <- FastRange(4)) {
          sum = Math.fma(this (y, k), m(k, x), sum)
        }
        result(y, x) = sum
      }
    }

    result
  }

  def multiplyNaiveFullyUnrolled(m: Matrix4x4): Matrix4x4 = {
    val result = new Matrix4x4()

    result(0, 0) = this (0, 0) * m(0, 0) + this (0, 1) * m(1, 0) + this (0, 2) * m(2, 0) + this (0, 3) * m(3, 0)
    result(0, 1) = this (0, 0) * m(0, 1) + this (0, 1) * m(1, 1) + this (0, 2) * m(2, 1) + this (0, 3) * m(3, 1)
    result(0, 2) = this (0, 0) * m(0, 2) + this (0, 1) * m(1, 2) + this (0, 2) * m(2, 2) + this (0, 3) * m(3, 2)
    result(0, 3) = this (0, 0) * m(0, 3) + this (0, 1) * m(1, 3) + this (0, 2) * m(2, 3) + this (0, 3) * m(3, 3)

    result(1, 0) = this (1, 0) * m(0, 0) + this (1, 1) * m(1, 0) + this (1, 2) * m(2, 0) + this (1, 3) * m(3, 0)
    result(1, 1) = this (1, 0) * m(0, 1) + this (1, 1) * m(1, 1) + this (1, 2) * m(2, 1) + this (1, 3) * m(3, 1)
    result(1, 2) = this (1, 0) * m(0, 2) + this (1, 1) * m(1, 2) + this (1, 2) * m(2, 2) + this (1, 3) * m(3, 2)
    result(1, 3) = this (1, 0) * m(0, 3) + this (1, 1) * m(1, 3) + this (1, 2) * m(2, 3) + this (1, 3) * m(3, 3)

    result(2, 0) = this (2, 0) * m(0, 0) + this (2, 1) * m(1, 0) + this (2, 2) * m(2, 0) + this (2, 3) * m(3, 0)
    result(2, 1) = this (2, 0) * m(0, 1) + this (2, 1) * m(1, 1) + this (2, 2) * m(2, 1) + this (2, 3) * m(3, 1)
    result(2, 2) = this (2, 0) * m(0, 2) + this (2, 1) * m(1, 2) + this (2, 2) * m(2, 2) + this (2, 3) * m(3, 2)
    result(2, 3) = this (2, 0) * m(0, 3) + this (2, 1) * m(1, 3) + this (2, 2) * m(2, 3) + this (2, 3) * m(3, 3)

    result(3, 0) = this (3, 0) * m(0, 0) + this (3, 1) * m(1, 0) + this (3, 2) * m(2, 0) + this (3, 3) * m(3, 0)
    result(3, 1) = this (3, 0) * m(0, 1) + this (3, 1) * m(1, 1) + this (3, 2) * m(2, 1) + this (3, 3) * m(3, 1)
    result(3, 2) = this (3, 0) * m(0, 2) + this (3, 1) * m(1, 2) + this (3, 2) * m(2, 2) + this (3, 3) * m(3, 2)
    result(3, 3) = this (3, 0) * m(0, 3) + this (3, 1) * m(1, 3) + this (3, 2) * m(2, 3) + this (3, 3) * m(3, 3)

    result
  }

  def multiplyFmaFullyUnrolled(m: Matrix4x4): Matrix4x4 = {
    val result = new Matrix4x4()

    result(0, 0) = Math.fma(this (0, 3), m(3, 0), Math.fma(this (0, 2), m(2, 0), Math.fma(this (0, 1), m(1, 0), this (0, 0) * m(0, 0))))
    result(0, 1) = Math.fma(this (0, 3), m(3, 1), Math.fma(this (0, 2), m(2, 1), Math.fma(this (0, 1), m(1, 1), this (0, 0) * m(0, 1))))
    result(0, 2) = Math.fma(this (0, 3), m(3, 2), Math.fma(this (0, 2), m(2, 2), Math.fma(this (0, 1), m(1, 2), this (0, 0) * m(0, 2))))
    result(0, 3) = Math.fma(this (0, 3), m(3, 3), Math.fma(this (0, 2), m(2, 3), Math.fma(this (0, 1), m(1, 3), this (0, 0) * m(0, 3))))

    result(1, 0) = Math.fma(this (1, 3), m(3, 0), Math.fma(this (1, 2), m(2, 0), Math.fma(this (1, 1), m(1, 0), this (1, 0) * m(0, 0))))
    result(1, 1) = Math.fma(this (1, 3), m(3, 1), Math.fma(this (1, 2), m(2, 1), Math.fma(this (1, 1), m(1, 1), this (1, 0) * m(0, 1))))
    result(1, 2) = Math.fma(this (1, 3), m(3, 2), Math.fma(this (1, 2), m(2, 2), Math.fma(this (1, 1), m(1, 2), this (1, 0) * m(0, 2))))
    result(1, 3) = Math.fma(this (1, 3), m(3, 3), Math.fma(this (1, 2), m(2, 3), Math.fma(this (1, 1), m(1, 3), this (1, 0) * m(0, 3))))

    result(2, 0) = Math.fma(this (2, 3), m(3, 0), Math.fma(this (2, 2), m(2, 0), Math.fma(this (2, 1), m(1, 0), this (2, 0) * m(0, 0))))
    result(2, 1) = Math.fma(this (2, 3), m(3, 1), Math.fma(this (2, 2), m(2, 1), Math.fma(this (2, 1), m(1, 1), this (2, 0) * m(0, 1))))
    result(2, 2) = Math.fma(this (2, 3), m(3, 2), Math.fma(this (2, 2), m(2, 2), Math.fma(this (2, 1), m(1, 2), this (2, 0) * m(0, 2))))
    result(2, 3) = Math.fma(this (2, 3), m(3, 3), Math.fma(this (2, 2), m(2, 3), Math.fma(this (2, 1), m(1, 3), this (2, 0) * m(0, 3))))

    result(3, 0) = Math.fma(this (3, 3), m(3, 0), Math.fma(this (3, 2), m(2, 0), Math.fma(this (3, 1), m(1, 0), this (3, 0) * m(0, 0))))
    result(3, 1) = Math.fma(this (3, 3), m(3, 1), Math.fma(this (3, 2), m(2, 1), Math.fma(this (3, 1), m(1, 1), this (3, 0) * m(0, 1))))
    result(3, 2) = Math.fma(this (3, 3), m(3, 2), Math.fma(this (3, 2), m(2, 2), Math.fma(this (3, 1), m(1, 2), this (3, 0) * m(0, 2))))
    result(3, 3) = Math.fma(this (3, 3), m(3, 3), Math.fma(this (3, 2), m(2, 3), Math.fma(this (3, 1), m(1, 3), this (3, 0) * m(0, 3))))

    result
  }

  private inline def loop(count: Int)(inline f: Int => Unit): Unit =
    var i = 0
    while (i < count) {
      f(i)
      i += 1
    }