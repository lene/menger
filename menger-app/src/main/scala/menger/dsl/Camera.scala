package menger.dsl

import menger.config.CameraConfig

/** Camera configuration for scene rendering.
  *
  * @param position Camera eye position in world space
  * @param lookAt Point the camera is looking at
  * @param up Camera up vector (typically (0, 1, 0))
  * @param fov Camera field of view in degrees — NOT YET IMPLEMENTED, must be None
  */
case class Camera(
  position: Vec3 = Vec3(0f, 0f, 3f),
  lookAt: Vec3 = Vec3.Zero,
  up: Vec3 = Vec3(0f, 1f, 0f),
  fov: Option[Float] = None
):
  fov.foreach(_ => failFov())

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def failFov(): Nothing =
    throw new NotImplementedError("fov is not yet implemented in the OptiX renderer")

  def toCameraConfig: CameraConfig =
    CameraConfig(
      position = position.toGdxVector3,
      lookAt = lookAt.toGdxVector3,
      up = up.toGdxVector3
    )

object Camera:
  /** Default camera: looking at origin from (0, 0, 3) */
  val Default: Camera = new Camera()
