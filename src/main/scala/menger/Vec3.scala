package menger

type Vec3[T] = (x: T, y: T, z: T)

object Vec3:
  val zero: Vec3[Int] = (x = 0, y = 0, z = 0)
