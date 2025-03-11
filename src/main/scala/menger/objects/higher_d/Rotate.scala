package menger.objects.higher_d

import com.badlogic.gdx.math.{Matrix4, Vector4}
import com.typesafe.scalalogging.LazyLogging

case class Rotate(transformationMatrix: Matrix4, pivotPoint: Vector4):
  /**
   In 3D the matrix of a proper rotation R by angle θ around the axis u = (ux, uy, uz),
   a unit vector, is given by:
   |    cos θ + ux^2(1 − cos θ)   uxuy(1 − cos θ) − uz sin θ   uxuz(1 − cos θ) + uy sin θ |
    R = | uyux(1 − cos θ) + uz sin θ   cos θ + uy^2(1 − cos θ)      uyuz(1 − cos θ) − ux sin θ |
   | uzux(1 − cos θ) − uy sin θ   uzuy(1 − cos θ) + ux sin θ   cos θ + uz^2(1 − cos θ)    |
  where cos θ is the cosine of the angle θ, sin θ is the sine of the angle θ,
  and ux, uy, and uz are the components of the unit vector u.

  This can be written more concisely as
   R = cos θ I + sin θ [u]_× + (1 − cos θ) (u ⊗ u),
   where [u]_× is the cross product matrix of u; the expression u ⊗ u is the outer product,
   and I is the identity matrix.
   | ux2  uxuy  uxuz |
   u ⊗ u = u uT = | uyux uy2   uyuz |
   | uzux uzuy  uz2  |
   The cross product matrix [u]_× is defined as
   |  0  −uz   uy |
   [u]_× = | uz    0  −ux |
   | −uy   ux   0 |
   (https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)

   template <> Rotation<4>::operator Matrix<4>() const {
   Matrix<4> Rxy = Matrix<4> (0, 1, axis[0]), Rxz = Matrix<4> (0, 2, axis[1]),
   Rxw = Matrix<4> (0, 3, axis[2]),  Ryz = Matrix<4> (1, 2, axis[3]),
   Ryw = Matrix<4> (1, 3, axis[4]), Rzw = Matrix<4> (2, 3, axis[5]),
   Rxyz = Rxy*Rxz, Rxwyz = Rxw*Ryz, Ryzw = Ryw*Rzw,
   Rot = Rxyz*Rxwyz*Ryzw;
   (https://github.com/lene/HyperspaceExplorer/blob/038b73b15e9462f015fb41a085ec6849ae0a6037/src/VecMath/Rotation.C#L59)

   template<unsigned D, typename N>
   Matrix<D, N>::Matrix (unsigned plane.i, unsigned jj, N theta) {
   N c = cos (theta*pi/180.), s = sin (theta*pi/180.);
   for (unsigned i = 0; i < D; i++) {          //  i: row
   for (unsigned j = 0; j < D; j++) {       //  j: col
   _M[i][j] = 0;
   }
   _M[i][i] = 1;
   }
   _M[ii][ii] =  _M[jj][jj] = c;
   _M[ii][jj] = -s;
   _M[jj][ii] = s;
   }
   (https://github.com/lene/HyperspaceExplorer/blob/038b73b15e9462f015fb41a085ec6849ae0a6037/src/VecMath/Matrix.impl.h#L44)

   Rotation of a point in 3 dimensional space by theta about an arbitrary axes defined by a line
   between two points P1 = (x1,y1,z1) and P2 = (x2,y2,z2) can be achieved by the following steps
   (1) translate space so that the rotation axis passes through the origin
   (2) rotate space about the x axis so that the rotation axis lies in the xz plane
   (3) rotate space about the y axis so that the rotation axis lies along the z axis
   (4) perform the desired rotation by theta about the z axis
   (5) apply the inverse of step (3)
   (6) apply the inverse of step (2)
   (7) apply the inverse of step (1)
   (https://paulbourke.net/geometry/rotate/)

   This can be shortened to:
   (1) translate the space so that the rotation axis passes through the origin
   (2) rotate about u axis as in the above formula
   (3) apply the inverse of step (1)
   */
  def apply(point: Vector4): Vector4 =
    transformationMatrix.multiply(point - pivotPoint) + pivotPoint

object Rotate extends LazyLogging:

  def apply(plane: Plane, axis: (Vector4, Vector4), pivotPoint: Vector4, angle: Float): Array[Rotate] =
    val u: Vector4 = axis(1) - axis(0)
    val direction: Int = u.toArray.indexWhere(math.abs(_) > Const.epsilon)
    val sign = math.signum(u.toArray(direction))
    val realAngle = sign * angle
    if direction != plane.i && direction != plane.j then
      throw new IllegalArgumentException(s"axis must be in the $plane plane, is $direction")
    logger.debug(s"from $plane around ${Seq('x', 'y', 'z', 'w')(direction)} at ${vec2string(pivotPoint)} by $realAngle°")
    val rotations: Array[Rotate] = plane match
      case Plane.xy =>
        direction match
          case 0 => plane.normalIndices.map(idx => apply(Plane(plane.j, idx), realAngle, pivotPoint))
          case 1 => plane.normalIndices.map(idx => apply(Plane(idx, plane.i), realAngle, pivotPoint))
          case _ => throw new IllegalArgumentException(s"axis must be in the $plane plane, is $direction")
      case Plane.xz =>
        direction match
          case 0 => plane.normalIndices.map(idx => apply(Plane(plane.j, idx), realAngle, pivotPoint))
          case 2 => plane.normalIndices.map(idx => apply(Plane(idx, plane.i), realAngle, pivotPoint))
          case _ => throw new IllegalArgumentException(s"axis must be in the $plane plane, is $direction")
      case Plane.xw =>
        direction match
          case 0 => plane.normalIndices.map(idx => apply(Plane(plane.j, idx), realAngle, pivotPoint))
          case 3 => plane.normalIndices.map(idx => apply(Plane(idx, plane.i), realAngle, pivotPoint))
          case _ => throw new IllegalArgumentException(s"axis must be in the $plane plane, is $direction")
      case Plane.yz =>
        direction match
          case 1 => plane.normalIndices.map(idx => apply(Plane(plane.j, idx), realAngle, pivotPoint))
          case 2 => plane.normalIndices.map(idx => apply(Plane(idx, plane.i), realAngle, pivotPoint))
          case _ => throw new IllegalArgumentException(s"axis must be in the $plane plane, is $direction")
      case Plane.yw =>
        direction match
          case 1 => plane.normalIndices.map(idx => apply(Plane(plane.j, idx), realAngle, pivotPoint))
          case 3 => plane.normalIndices.map(idx => apply(Plane(idx, plane.i), realAngle, pivotPoint))
          case _ => throw new IllegalArgumentException(s"axis must be in the $plane plane, is $direction")
      case Plane.zw =>
        direction match
          case 2 => plane.normalIndices.map(idx => apply(Plane(plane.j, idx), realAngle, pivotPoint))
          case 3 => plane.normalIndices.map(idx => apply(Plane(idx, plane.i), realAngle, pivotPoint))
          case _ => throw new IllegalArgumentException(s"axis must be in the $plane plane, is $direction")
      case _ =>
        throw new IllegalArgumentException(s"plane must be xy, xz, xw, yz, yw, or zw, is $plane")
    rotations

  def apply(plane: Plane, angle: Float, pivotPoint: Vector4 = Vector4.Zero): Rotate =
    Rotate(transformationMatrix(plane, angle), pivotPoint)

  private def transformationMatrix(plane: Plane, angle: Float) =
    val cosTheta: Float = math.cos(angle.toRadians).toFloat
    val sinTheta: Float = math.sin(angle.toRadians).toFloat
    val values: Array[Float] = Array.fill(16)(0)
    for i <- 0 to 3 do values(i * 4 + i) = 1
    values(plane.i * 4 + plane.i) = cosTheta
    values(plane.j * 4 + plane.j) = cosTheta
    values(plane.i * 4 + plane.j) = -sinTheta
    values(plane.j * 4 + plane.i) = sinTheta
    Matrix4(values)

  private def outerProduct(u: Vector4): Matrix4 =
    val (x, y, z, w) = (u.x, u.y, u.z, u.w)
    Matrix4(Array(
      x * x, x * y, x * z, x * w,
      y * x, y * y, y * z, y * w,
      z * x, z * y, z * z, z * w,
      w * x, w * y, w * z, w * w
    ))

  private def crossProductMatrix(u: Vector4): Matrix4 =
    val (x, y, z, w) = (u.x, u.y, u.z, u.w)
    Matrix4(Array(
      0, -z, y, 0,
      z, 0, -x, 0,
      -y, x, 0, 0,
      0, 0, 0, 0
    ))
