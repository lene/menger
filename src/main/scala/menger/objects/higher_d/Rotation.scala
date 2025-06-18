package menger.objects.higher_d

import com.badlogic.gdx.math.{Matrix4, Vector4}
import com.typesafe.scalalogging.LazyLogging
import menger.{Const, RotationProjectionParameters}

import scala.annotation.targetName

case class Rotation(transformationMatrix: Matrix4, pivotPoint: Vector4) extends RectMesh:
  lazy val isZero: Boolean = epsilonEquals(transformationMatrix, Matrix4())

  def apply(point: Vector4): Vector4 =
    if isZero then point else transformationMatrix(point - pivotPoint) + pivotPoint

  def apply(points: Seq[Vector4]): Seq[Vector4] = if isZero then points else points.map(apply)

  def apply(points: Face4D): Face4D = Face4D(apply(points.asSeq))

  @targetName("plus")
  def +(r: Rotation): Rotation = Rotation(transformationMatrix.mul(r.transformationMatrix), pivotPoint)

  @targetName("mul")
  def *(r: Rotation): Rotation = Rotation(transformationMatrix.mul(r.transformationMatrix), pivotPoint)

  def ===(r: Rotation): Boolean =
    epsilonEquals(transformationMatrix, r.transformationMatrix) && pivotPoint === r.pivotPoint

object Rotation extends LazyLogging:

  def apply(): Rotation = Rotation(Matrix4(), Vector4.Zero)

  def apply(
    degreesXW: Float, degreesYW: Float, degreesZW: Float, pivotPoint: Vector4 = Vector4.Zero
  ): Rotation =
    val Ryw = Rotation.matrix(1, 3, degreesYW)
    val Rzw = Rotation.matrix(2, 3, degreesZW)
    val rotate = Rotation.matrix(0, 3, degreesXW).mul(Ryw).mul(Rzw)
    Rotation(rotate, pivotPoint)

  def apply(rotProjParameters: RotationProjectionParameters): Rotation =
    val Rx = Rotation.matrix(0, 1, rotProjParameters.rotX)
    val Ry = Rotation.matrix(0, 2, rotProjParameters.rotY)
    val rot3D = Rotation.matrix(0, 0, rotProjParameters.rotZ).mul(Rx).mul(Ry)
    Rotation(rotProjParameters.rotXW, rotProjParameters.rotYW, rotProjParameters.rotZW) * Rotation(rot3D, Vector4.Zero)

  def apply(plane: Plane, axis: Edge, pivotPoint: Vector4, angle: Float): Array[Rotation] =
    val u: Vector4 = axis(1) - axis(0)
    val direction: Int = u.toArray.indexWhere(math.abs(_) > Const.epsilon)
    require(
      direction == plane.i || direction == plane.j,
      s"axis must be in the $plane plane, is $direction"
    )
    val sign = math.signum(u.toArray(direction))
    val realAngle = sign * angle
    logger.debug(s"from $plane around ${Seq('x', 'y', 'z', 'w')(direction)} at ${vec2string(pivotPoint)} by $realAngle°")
    direction match
      case plane.i => plane.normalIndices.map(idx => apply(Plane(plane.j, idx), realAngle, pivotPoint))
      case plane.j => plane.normalIndices.map(idx => apply(Plane(idx, plane.i), realAngle, pivotPoint))

  def apply(plane: Plane, angle: Float, pivotPoint: Vector4): Rotation =
    Rotation(matrix(plane, angle), pivotPoint)

  def apply(plane: Plane, angle: Float): Rotation =
    Rotation(matrix(plane, angle), Vector4.Zero)

  private def matrix(row: Int, col: Int, angle: Float): Matrix4 =
    val cosTheta: Float = math.cos(angle.toRadians).toFloat
    val sinTheta: Float = math.sin(angle.toRadians).toFloat
    val m = Array.fill(4, 4)(0f)
    for i <- 0 to 3 do m(i)(i) = 1f
    m(row)(row) = cosTheta
    m(row)(col) = -sinTheta
    m(col)(row) = sinTheta
    m(col)(col) = cosTheta
    Matrix4(m.flatten)

  private def matrix(plane: Plane, angle: Float): Matrix4 = Rotation.matrix(plane.i, plane.j, angle)

def epsilonEquals(m1: Matrix4, m2: Matrix4): Boolean =
  m2.asArray.zip(m1.asArray).forall((a, b) => math.abs(a - b) < Const.epsilon)
