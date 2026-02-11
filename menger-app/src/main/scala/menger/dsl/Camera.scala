package menger.dsl

import scala.annotation.targetName

import menger.config.CameraConfig

/** Camera configuration for scene rendering.
  *
  * @param position Camera eye position in world space
  * @param lookAt Point the camera is looking at
  * @param up Camera up vector (typically (0, 1, 0))
  */
case class Camera(
  position: Vec3 = Vec3(0f, 0f, 3f),
  lookAt: Vec3 = Vec3.Zero,
  up: Vec3 = Vec3(0f, 1f, 0f)
):
  def toCameraConfig: CameraConfig =
    CameraConfig(
      position = position.toGdxVector3,
      lookAt = lookAt.toGdxVector3,
      up = up.toGdxVector3
    )

object Camera:
  /** Default camera: looking at origin from (0, 0, 3) */
  val Default: Camera = new Camera()

  // Float tuple overloads for position
  @targetName("applyFloatTupleLookAtTuple")
  def apply(position: (Float, Float, Float), lookAt: (Float, Float, Float)): Camera =
    new Camera(
      Vec3(position._1, position._2, position._3),
      Vec3(lookAt._1, lookAt._2, lookAt._3)
    )

  @targetName("applyFloatTupleLookAtTupleUpTuple")
  def apply(
    position: (Float, Float, Float),
    lookAt: (Float, Float, Float),
    up: (Float, Float, Float)
  ): Camera =
    new Camera(
      Vec3(position._1, position._2, position._3),
      Vec3(lookAt._1, lookAt._2, lookAt._3),
      Vec3(up._1, up._2, up._3)
    )

  // Int tuple overloads
  @targetName("applyIntTupleLookAtTuple")
  def apply(position: (Int, Int, Int), lookAt: (Int, Int, Int)): Camera =
    new Camera(
      Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat),
      Vec3(lookAt._1.toFloat, lookAt._2.toFloat, lookAt._3.toFloat)
    )

  // Double tuple overloads
  @targetName("applyDoubleTupleLookAtTuple")
  def apply(position: (Double, Double, Double), lookAt: (Double, Double, Double)): Camera =
    new Camera(
      Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat),
      Vec3(lookAt._1.toFloat, lookAt._2.toFloat, lookAt._3.toFloat)
    )
