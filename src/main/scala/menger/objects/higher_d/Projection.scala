package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import menger.objects.Vector

import scala.annotation.targetName
import scala.math.{pow, signum}


/** project 4D points to 3D where the point we look at is at the origin `(0, 0, 0, 0)`,
 *  the eye is at `(0, 0, 0, -eyeW)` and the screen is at `(0, 0, 0, -screenW)`
 */
case class Projection(eyeW: Float, screenW: Float) extends RectMesh:
  require(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")
  require(eyeW > screenW, "eyeW must be greater than screenW")

  final val addExponent = 1.1

  @targetName("plus")
  def +(p: Projection): Projection = Projection(newEyeW(p), screenW)

  private def newEyeW(p: Projection): Float = pow(eyeW, exponent(p)).toFloat
  private def exponent(p: Projection): Double = pow(addExponent, signum(p.eyeW - eyeW))

  /** project a single 4D point `point` to 3D */
  def apply(point: Vector[4]): Vector3 =
    val projectionFactor = (eyeW - screenW) / (eyeW - point(3))
    Vector3(point(0) * projectionFactor, point(1) * projectionFactor, point(2) * projectionFactor)

  /** project a sequence of 4D points to 3D */
  def apply(points: Seq[Vector[4]]): Seq[Vector3] = points.map(apply)
  def apply(points: Face4D): Quad3D =
    val vectors: Seq[Vector3] = apply(points.asSeq)
    Quad3D(vectors(0), vectors(1), vectors(2), vectors(3))
