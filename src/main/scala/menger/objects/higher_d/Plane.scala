package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4

case class Plane(i: Int, j: Int):
  require(i >= 0 && i < 4 && j >= 0 && j < 4, s"i and j must be between 0 and 3, are $i and $j")

object Plane:
  val xy: Plane = Plane(0, 1)
  val xz: Plane = Plane(0, 2)
  val xw: Plane = Plane(0, 3)
  val yz: Plane = Plane(1, 2)
  val yw: Plane = Plane(1, 3)
  val zw: Plane = Plane(2, 3)

  def apply(cornerPoints: Array[Vector4]): Plane =
    val setIndices = cornerPoints.foldLeft(Set.empty[Int])((set, v) => set + v.toArray.indexWhere(_ != 0))
    if setIndices.size != 2 then throw new IllegalArgumentException(s"Corner points must lie in a plane, has $setIndices (${cornerPoints.mkString(", ")})")
    Plane(setIndices.head, setIndices.last)

  def apply(cornerPoints: (Vector4, Vector4, Vector4, Vector4)): Plane =
    apply(Array(cornerPoints._1, cornerPoints._2, cornerPoints._3, cornerPoints._4))