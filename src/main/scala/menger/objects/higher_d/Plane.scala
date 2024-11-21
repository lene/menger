package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4

import scala.::

case class Plane(i: Int, j: Int):
  require(i >= 0 && i < 4 && j >= 0 && j < 4, s"i and j must be between 0 and 3, are $i and $j")
  lazy val indices: Array[Int] = Array(i, j)
  lazy val normalIndices: Array[Int] = Set(0, 1, 2, 3).diff(indices.toSet).toArray
  override def toString: String = s"${("x", "y", "z", "w")(i)}${("x", "y", "z", "w")(j)}"

object Plane:
  val epsilon: Float = 1e-6f
  val xy: Plane = Plane(0, 1)
  val xz: Plane = Plane(0, 2)
  val xw: Plane = Plane(0, 3)
  val yz: Plane = Plane(1, 2)
  val yw: Plane = Plane(1, 3)
  val zw: Plane = Plane(2, 3)

  def apply(cornerPoints: Seq[Vector4]): Plane =
    if cornerPoints.isEmpty then throw new IllegalArgumentException("Corner points must not be empty")
    val setIndices = Plane.setIndices(cornerPoints)
    if setIndices.length != 2 then throw new IllegalArgumentException(
      s"Corner points must lie in a plane, has ${setIndices.mkString(", ")} (${cornerPoints.mkString(", ")})"
    )
    Plane(setIndices.head, setIndices.last)

  def differenceVectors(cornerPoints: Seq[Vector4]): Seq[Vector4] =
    differences(cornerPoints :+ cornerPoints.head)

  def differences(seq: Seq[Vector4]): Seq[Vector4] =
    seq.sliding(2).collect { case Seq(a, b) => b - a }.toSeq

  def setIndices(cornerPoints: Seq[Vector4]): Array[Int] =
    differenceVectors(cornerPoints).foldLeft(Set.empty[Int])((set, v) => set + v.toArray.indexWhere(math.abs(_) > epsilon)).toArray.sorted

  def apply(cornerPoints: (Vector4, Vector4, Vector4, Vector4)): Plane =
    apply(Seq(cornerPoints._1, cornerPoints._2, cornerPoints._3, cornerPoints._4))
