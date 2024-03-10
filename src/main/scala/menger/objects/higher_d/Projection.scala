package menger.objects.higher_d

import com.badlogic.gdx.math.{Vector3, Vector4}


/** project 4D points to 3D where the point we look at is at the origin `(0, 0, 0, 0)`,
 *  the eye is at `(0, 0, 0, -eyeW)` and the screen is at `(0, 0, 0, -screenW)`
 */
case class Projection(eyeW: Float, screenW: Float) extends RectMesh:
  assert(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")
  assert(eyeW > screenW, "eyeW must be greater than screenW")

  /** project a single 4D point `point` to 3D */
  def apply(point: Vector4): Vector3 =
    val projectionFactor = (eyeW - screenW) / (eyeW - point.w)
    Vector3(point.x * projectionFactor, point.y * projectionFactor, point.z * projectionFactor)

  /** project a sequence of 4D points to 3D */
  def apply(points: Seq[Vector4]): Seq[Vector3] = points.map(apply)
  def apply(points: RectVertices4D): RectVertices3D = (
    apply(points(0)), apply(points(1)), apply(points(2)), apply(points(3))
  )
