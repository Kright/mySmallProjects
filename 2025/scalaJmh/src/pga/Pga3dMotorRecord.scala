package pga

final case class Pga3dMotorRecord(s: Double = 0.0,
                                  wx: Double = 0.0,
                                  wy: Double = 0.0,
                                  wz: Double = 0.0,
                                  xy: Double = 0.0,
                                  xz: Double = 0.0,
                                  yz: Double = 0.0,
                                  i: Double = 0.0) extends Record:

  infix def geometric(v: Pga3dMotorRecord): Pga3dMotorRecord =
    Pga3dMotorRecord(
      s = (s * v.s - v.xy * xy - v.xz * xz - v.yz * yz),
      wx = (s * v.wx + v.s * wx + v.wy * xy + v.wz * xz - i * v.yz - v.i * yz - v.xy * wy - v.xz * wz),
      wy = (i * v.xz + s * v.wy + v.i * xz + v.s * wy + v.wz * yz + v.xy * wx - v.wx * xy - v.yz * wz),
      wz = (s * v.wz + v.s * wz + v.xz * wx + v.yz * wy - i * v.xy - v.i * xy - v.wx * xz - v.wy * yz),
      xy = (s * v.xy + v.s * xy + v.xz * yz - v.yz * xz),
      xz = (s * v.xz + v.s * xz + v.yz * xy - v.xy * yz),
      yz = (s * v.yz + v.s * yz + v.xy * xz - v.xz * xy),
      i = (i * v.s + s * v.i + v.wx * yz + v.wz * xy + v.xy * wz + v.yz * wx - v.wy * xz - v.xz * wy),
    )

    