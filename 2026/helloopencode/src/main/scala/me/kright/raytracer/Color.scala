package me.kright.raytracer

case class Color(r: Double, g: Double, b: Double):
  inline def +(c: Color): Color = Color(r + c.r, g + c.g, b + c.b)
  inline def *(t: Double): Color = Color(r * t, g * t, b * t)
  inline def *(c: Color): Color = Color(r * c.r, g * c.g, b * c.b)
  inline def /(t: Double): Color = Color(r / t, g / t, b / t)

  def toRGB: (Int, Int, Int) =
    def gammaCorrect(x: Double): Int =
      (math.pow(x.max(0.0).min(0.999), 1.0 / 2.2) * 255.999).toInt
    (gammaCorrect(r), gammaCorrect(g), gammaCorrect(b))
