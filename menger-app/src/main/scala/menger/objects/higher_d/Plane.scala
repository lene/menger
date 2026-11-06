package menger.objects.higher_d

import menger.common.Const
import menger.common.Vector


case class Plane(i: Int, j: Int, dimension: Int = Plane.legacyDimension):
  Dimension.requireSupported(dimension, s"plane ($i, $j)")
  require(
    i >= 0 && i < dimension && j >= 0 && j < dimension,
    s"i and j must be between 0 and ${dimension - 1} in dimension $dimension, are $i and $j"
  )
  require(i != j, s"i and j must differ, are $i and $j in dimension $dimension")
  lazy val indices: Array[Int] = Array(i, j)
  lazy val normalIndices: Array[Int] =
    (0 until dimension).filterNot(indices.contains).toArray
  override def toString: String = s"${Dimension.axisNames(i)}${Dimension.axisNames(j)}"
  def neg: Plane =
    requireLegacyDimension("neg")
    val unusedIndices = (0 to 3).diff(indices)
    Plane(unusedIndices.head, unusedIndices.last)
  def units: Seq[Vector[4]] =
    requireLegacyDimension("units")
    Seq(Plane.units(i), Plane.units(j))

  private def requireLegacyDimension(operation: String): Unit =
    require(
      dimension == Plane.legacyDimension,
      s"$operation is only defined for 4D planes, plane $this is in dimension $dimension"
    )

object Plane:
  val legacyDimension: Int = 4
  val xy: Plane = Plane(0, 1)
  val xz: Plane = Plane(0, 2)
  val xw: Plane = Plane(0, 3)
  val yz: Plane = Plane(1, 2)
  val yw: Plane = Plane(1, 3)
  val zw: Plane = Plane(2, 3)
  val units: Array[Vector[4]] = Array(
    Vector.X, Vector.Y, Vector.Z, Vector.W
  )

  def apply(cornerPoints: Seq[Vector[4]]): Plane =
    require(cornerPoints.nonEmpty, "Corner points must not be empty")
    require(cornerPoints.length >= 3, s"Need at least 3 corner points, have $cornerPoints")
    val setIndices = Plane.setIndices(cornerPoints)
    require(
      setIndices.length == 2, s"Corner points must lie in a plane, are $setIndices ($cornerPoints)"
    )
    Plane(setIndices.head, setIndices.last)

  def apply(cornerPoints: Face4D[?]): Plane = apply(cornerPoints.asSeq)

  def differenceVectors(cornerPoints: Seq[Vector[4]]): Seq[Vector[4]] =
    differences(cornerPoints :+ cornerPoints.head)

  private def differences(seq: Seq[Vector[4]]): Seq[Vector[4]] =
    seq.sliding(2).collect { case Seq(a, b) => b - a }.toSeq

  def setIndices(cornerPoints: Seq[Vector[4]]): Array[Int] =
    differenceVectors(cornerPoints).foldLeft(
      Set.empty[Int])((set, v) => set + v.indexWhere(math.abs(_) > Const.epsilon)
    ).toArray.sorted
