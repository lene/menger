package menger.common

extension (v: Vector[3])
  def x: Float = v(0)
  def y: Float = v(1)
  def z: Float = v(2)
  def toArray: Array[Float] = v.v.toArray

object Vector3:
  def fromArray(arr: Array[Float]): Vector[3] =
    require(arr.length == 3, s"Expected 3 elements, got ${arr.length}")
    Vector.fromSeq[3](arr.toSeq)
