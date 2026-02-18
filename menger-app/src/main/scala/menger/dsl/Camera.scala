package menger.dsl

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
