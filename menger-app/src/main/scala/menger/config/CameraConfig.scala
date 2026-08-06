package menger.config

import menger.common.Vector

/**
 * Camera configuration for rendering.
 *
 * @param position camera eye position in world space
 * @param lookAt point the camera is looking at
 * @param up camera up vector (typically (0, 1, 0))
 */
case class CameraConfig(
  position: Vector[3],
  lookAt: Vector[3],
  up: Vector[3]
)

object CameraConfig:
  /**
   * Default camera configuration: looking at origin from (0, 0, 3)
   */
  val Default: CameraConfig = CameraConfig(
    position = Vector[3](0f, 0f, 3f),
    lookAt = Vector[3](0f, 0f, 0f),
    up = Vector[3](0f, 1f, 0f)
  )
