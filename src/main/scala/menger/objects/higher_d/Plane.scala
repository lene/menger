package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import com.typesafe.scalalogging.Logger

val logger = Logger("menger.objects.higher_d.Plane")

case class Plane(i: Int, j: Int):
  require(i >= 0 && i < 4 && j >= 0 && j < 4, s"i and j must be between 0 and 3, are $i and $j")
  lazy val indices: Array[Int] = Array(i, j)
  lazy val normalIndices: Array[Int] = Set(0, 1, 2, 3).diff(indices.toSet).toArray
  override def toString: String = s"${("x", "y", "z", "w")(i)}${("x", "y", "z", "w")(j)}"
  def neg: Plane =
    val unusedIndices = (0 to 3).diff(indices)
    Plane(unusedIndices.head, unusedIndices.last)
  def units: Seq[Vector4] = Seq(Plane.units(i), Plane.units(j)) 

object Plane:
  val epsilon: Float = 1e-6f
  val xy: Plane = Plane(0, 1)
  val xz: Plane = Plane(0, 2)
  val xw: Plane = Plane(0, 3)
  val yz: Plane = Plane(1, 2)
  val yw: Plane = Plane(1, 3)
  val zw: Plane = Plane(2, 3)
  val units: Array[Vector4] = Array(
    Vector4.X, Vector4.Y, Vector4.Z, Vector4.W
  )

  def apply(cornerPoints: Seq[Vector4]): Plane =
    require(cornerPoints.nonEmpty, "Corner points must not be empty")
    require(cornerPoints.length >= 3, s"Need at least 3 corner points, have $cornerPoints")
    val setIndices = Plane.setIndices(cornerPoints)
    require(
      setIndices.length == 2,
      s"Corner points must lie in a plane, has ${setIndices.mkString(", ")} (${cornerPoints.mkString(", ")})"
    )
    val differences: Array[Vector4] = cornerPoints.foldLeft(Array(cornerPoints.last - cornerPoints.head))((diffs, v) => diffs :+ v - diffs.head)
    logger.debug(
      s"cornerPoints: ${cornerPoints.mkString(", ")}\n" +
      s"differences: ${differences.mkString(", ")}\n" +
      s"diffs: ${differences.map(_.toArray.mkString(", ")).mkString("\n")}"
    )
    Plane(setIndices.head, setIndices.last)

  def apply(cornerPoints: Face4D): Plane = apply(cornerPoints.asSeq)

  def differenceVectors(cornerPoints: Seq[Vector4]): Seq[Vector4] =
    differences(cornerPoints :+ cornerPoints.head)

  def differences(seq: Seq[Vector4]): Seq[Vector4] =
    seq.sliding(2).collect { case Seq(a, b) => b - a }.toSeq

  def setIndices(cornerPoints: Seq[Vector4]): Array[Int] =
    differenceVectors(cornerPoints).foldLeft(
      Set.empty[Int])((set, v) => set + v.toArray.indexWhere(math.abs(_) > epsilon)
    ).toArray.sorted
