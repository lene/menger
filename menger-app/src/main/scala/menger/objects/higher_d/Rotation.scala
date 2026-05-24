package menger.objects.higher_d

import scala.annotation.targetName

import menger.RotationProjectionParameters
import menger.common.Const
import menger.common.Vector
import menger.objects.Matrix

case class Rotation(transformationMatrix: Matrix[4], pivotPoint: Vector[4]) extends RectMesh:
  lazy val isZero: Boolean = transformationMatrix === Matrix.identity[4]

  def apply(point: Vector[4]): Vector[4] =
    if isZero then point else transformationMatrix(point - pivotPoint) + pivotPoint

  def apply(points: Seq[Vector[4]]): Seq[Vector[4]] = if isZero then points else points.map(apply)

  def apply[V <: Int & Singleton](points: Face4D[V])(using ValueOf[V]): Face4D[V] =
    Face4D[V](points.vertices.map(apply))

  @targetName("plus")
  def +(r: Rotation): Rotation = Rotation(transformationMatrix * r.transformationMatrix, pivotPoint)

  @targetName("mul")
  def *(r: Rotation): Rotation = Rotation(transformationMatrix * r.transformationMatrix, pivotPoint)

  def ===(r: Rotation): Boolean =
    transformationMatrix === r.transformationMatrix && pivotPoint === r.pivotPoint

object Rotation:

  val identity: Rotation = Rotation(Matrix.identity[4], Vector.Zero[4])

  def apply(): Rotation = identity

  def apply(
    degreesXW: Float, degreesYW: Float, degreesZW: Float, pivotPoint: Vector[4] = Vector.Zero[4]
  ): Rotation =
    val Ryw = Rotation.matrix(1, 3, degreesYW)
    val Rzw = Rotation.matrix(2, 3, degreesZW)
    val rotate = Rotation.matrix(0, 3, degreesXW) * Ryw * Rzw
    Rotation(rotate, pivotPoint)

  def apply(rotProjParameters: RotationProjectionParameters): Rotation =
    val Rx = Rotation.matrix(0, 1, rotProjParameters.rotX)
    val Ry = Rotation.matrix(0, 2, rotProjParameters.rotY)
    val rot3D = Rotation.matrix(0, 0, rotProjParameters.rotZ) * Rx * Ry
    Rotation(rotProjParameters.rotXW, rotProjParameters.rotYW, rotProjParameters.rotZW) * Rotation(rot3D, Vector.Zero[4])

  def apply(plane: Plane, axis: Edge, pivotPoint: Vector[4], angle: Float): Array[Rotation] =
    val u: Vector[4] = axis(1) - axis(0)
    val direction: Int = u.indexWhere(math.abs(_) > Const.epsilon)
    require(
      direction == plane.i || direction == plane.j,
      s"axis must be in the $plane plane, is $direction"
    )
    val sign = math.signum(u(direction))
    val realAngle = sign * angle
    // rotation: from plane around axis at pivotPoint by realAngle degrees
    direction match
      case plane.i => plane.normalIndices.map(idx => apply(Plane(plane.j, idx), realAngle, pivotPoint))
      case plane.j => plane.normalIndices.map(idx => apply(Plane(idx, plane.i), realAngle, pivotPoint))

  def apply(plane: Plane, angle: Float, pivotPoint: Vector[4]): Rotation =
    Rotation(matrix(plane, angle), pivotPoint)

  def apply(plane: Plane, angle: Float): Rotation =
    Rotation(matrix(plane, angle), Vector.Zero[4])

  private def matrix(row: Int, col: Int, angle: Float): Matrix[4] =
    val cosTheta: Float = math.cos(angle.toRadians).toFloat
    val sinTheta: Float = math.sin(angle.toRadians).toFloat
    val m = Array.fill(4, 4)(0f)
    for i <- 0 to 3 do m(i)(i) = 1f
    m(row)(row) = cosTheta
    m(col)(row) = -sinTheta
    m(row)(col) = sinTheta
    m(col)(col) = cosTheta
    Matrix[4](m.flatten)

  private def matrix(plane: Plane, angle: Float): Matrix[4] = Rotation.matrix(plane.i, plane.j, angle)
