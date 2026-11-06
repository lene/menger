package menger.objects.higher_d

import scala.annotation.targetName

import menger.common.Const
import menger.common.Vector
import menger.objects.Matrix

case class Rotation[N <: Int & Singleton](
  transformationMatrix: Matrix[N], pivotPoint: Vector[N]
)(using ValueOf[N]):
  lazy val isZero: Boolean = transformationMatrix === Matrix.identity[N]

  def apply(point: Vector[N]): Vector[N] =
    if isZero then point else transformationMatrix(point - pivotPoint) + pivotPoint

  def apply(points: Seq[Vector[N]]): Seq[Vector[N]] = if isZero then points else points.map(apply)

  def apply[V <: Int & Singleton](points: Polygon[N, V])(using ValueOf[V]): Polygon[N, V] =
    Polygon[N, V](points.vertices.map(apply))

  @targetName("plus")
  def +(r: Rotation[N]): Rotation[N] =
    Rotation[N](transformationMatrix * r.transformationMatrix, pivotPoint)

  @targetName("mul")
  def *(r: Rotation[N]): Rotation[N] =
    Rotation[N](transformationMatrix * r.transformationMatrix, pivotPoint)

  def ===(r: Rotation[N]): Boolean =
    transformationMatrix === r.transformationMatrix && pivotPoint === r.pivotPoint

object Rotation:

  val identity: Rotation[4] = Rotation[4](Matrix.identity[4], Vector.Zero[4])

  def apply(): Rotation[4] = identity

  def apply(
    degreesXW: Float, degreesYW: Float, degreesZW: Float, pivotPoint: Vector[4] = Vector.Zero[4]
  ): Rotation[4] =
    val Ryw = planeMatrix[4](Plane.yw, degreesYW)
    val Rzw = planeMatrix[4](Plane.zw, degreesZW)
    val rotate = planeMatrix[4](Plane.xw, degreesXW) * Ryw * Rzw
    Rotation[4](rotate, pivotPoint)

  def apply(
    plane: Plane, axis: Edge, pivotPoint: Vector[4], angle: Float
  ): Array[Rotation[4]] =
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

  def apply(plane: Plane, angle: Float, pivotPoint: Vector[4]): Rotation[4] =
    Rotation[4](planeMatrix[4](plane, angle), pivotPoint)

  def apply(plane: Plane, angle: Float): Rotation[4] =
    Rotation[4](planeMatrix[4](plane, angle), Vector.Zero[4])

  def inPlane[N <: Int & Singleton](plane: Plane, degrees: Float)(using ValueOf[N]): Rotation[N] =
    inPlane[N](plane, degrees, Vector.Zero[N])

  def inPlane[N <: Int & Singleton](
    plane: Plane, degrees: Float, pivotPoint: Vector[N]
  )(using ValueOf[N]): Rotation[N] =
    Rotation[N](planeMatrix[N](plane, degrees), pivotPoint)

  // The first turn in `turns` is applied to a point first. Composed without an identity factor:
  // multiplying by one can flip the sign of a zero.
  def inWrittenOrder[N <: Int & Singleton](
    turns: Seq[(Plane, Float)]
  )(using ValueOf[N]): Rotation[N] =
    inWrittenOrder[N](turns, Vector.Zero[N])

  def inWrittenOrder[N <: Int & Singleton](
    turns: Seq[(Plane, Float)], pivotPoint: Vector[N]
  )(using ValueOf[N]): Rotation[N] =
    val matrices = turns.map((plane, degrees) => planeMatrix[N](plane, degrees))
    val composed = matrices.reduceLeftOption((first, next) => next * first)
    Rotation[N](composed.getOrElse(Matrix.identity[N]), pivotPoint)

  def planeMatrix[N <: Int & Singleton](plane: Plane, angle: Float)(
    using d: ValueOf[N]
  ): Matrix[N] =
    val size = d.value
    require(Dimension.isSupported(size), s"dimension must be between ${Dimension.min} and " +
      s"${Dimension.max}, is $size (plane $plane)")
    require(plane.dimension == size,
      s"plane $plane is in dimension ${plane.dimension}, not in the rotation's dimension $size")
    val cosTheta: Float = math.cos(angle.toRadians).toFloat
    val sinTheta: Float = math.sin(angle.toRadians).toFloat
    val m = Array.fill(size * size)(0f)
    for k <- 0 until size do m(k * size + k) = 1f
    m(plane.i * size + plane.i) = cosTheta
    m(plane.j * size + plane.i) = -sinTheta
    m(plane.i * size + plane.j) = sinTheta
    m(plane.j * size + plane.j) = cosTheta
    Matrix[N](m)
