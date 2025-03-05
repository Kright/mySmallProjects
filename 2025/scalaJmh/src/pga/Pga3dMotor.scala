package pga

final case class Pga3dMotor(s: Double = 0.0,
                            wx: Double = 0.0,
                            wy: Double = 0.0,
                            wz: Double = 0.0,
                            xy: Double = 0.0,
                            xz: Double = 0.0,
                            yz: Double = 0.0,
                            i: Double = 0.0):

  infix def geometric(v: Pga3dMotor): Pga3dMotor =
    Pga3dMotor(
      s = (s * v.s - v.xy * xy - v.xz * xz - v.yz * yz),
      wx = (s * v.wx + v.s * wx + v.wy * xy + v.wz * xz - i * v.yz - v.i * yz - v.xy * wy - v.xz * wz),
      wy = (i * v.xz + s * v.wy + v.i * xz + v.s * wy + v.wz * yz + v.xy * wx - v.wx * xy - v.yz * wz),
      wz = (s * v.wz + v.s * wz + v.xz * wx + v.yz * wy - i * v.xy - v.i * xy - v.wx * xz - v.wy * yz),
      xy = (s * v.xy + v.s * xy + v.xz * yz - v.yz * xz),
      xz = (s * v.xz + v.s * xz + v.yz * xy - v.xy * yz),
      yz = (s * v.yz + v.s * yz + v.xy * xz - v.xz * xy),
      i = (i * v.s + s * v.i + v.wx * yz + v.wz * xy + v.xy * wz + v.yz * wx - v.wy * xz - v.xz * wy),
    )

  infix def geometricWithFma(v: Pga3dMotor): Pga3dMotor =
    Pga3dMotor(
      s = Math.fma(s, v.s, Math.fma(-v.xy, xy, Math.fma(-v.xz, xz, -v.yz * yz))),
      wx = Math.fma(s, v.wx, Math.fma(v.s, wx, Math.fma(v.wy, xy, Math.fma(v.wz, xz, Math.fma(-i, v.yz, Math.fma(-v.i, yz, Math.fma(-v.xy, wy, -v.xz * wz))))))),
      wy = Math.fma(i, v.xz, Math.fma(s, v.wy, Math.fma(v.i, xz, Math.fma(v.s, wy, Math.fma(v.wz, yz, Math.fma(v.xy, wx, Math.fma(-v.wx, xy, -v.yz * wz))))))),
      wz = Math.fma(s, v.wz, Math.fma(v.s, wz, Math.fma(v.xz, wx, Math.fma(v.yz, wy, Math.fma(-i, v.xy, Math.fma(-v.i, xy, Math.fma(-v.wx, xz, -v.wy * yz))))))),
      xy = Math.fma(s, v.xy, Math.fma(v.s, xy, Math.fma(v.xz, yz, -v.yz * xz))),
      xz = Math.fma(s, v.xz, Math.fma(v.s, xz, Math.fma(v.yz, xy, -v.xy * yz))),
      yz = Math.fma(s, v.yz, Math.fma(v.s, yz, Math.fma(v.xy, xz, -v.xz * xy))),
      i = Math.fma(i, v.s, Math.fma(s, v.i, Math.fma(v.wx, yz, Math.fma(v.wz, xy, Math.fma(v.xy, wz, Math.fma(v.yz, wx, Math.fma(-v.wy, xz, -v.xz * wy)))))))
    )
