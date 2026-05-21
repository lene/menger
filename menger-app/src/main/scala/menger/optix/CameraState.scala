package menger.optix

import com.typesafe.scalalogging.LazyLogging
import menger.common.ImageSize
import menger.common.Vector

class CameraState(
  initialPos: Vector[3],
  initialLookat: Vector[3],
  initialUp: Vector[3]
) extends LazyLogging:

  private val horizontalFov = 45f  // Fixed horizontal FOV in degrees (aspect-ratio independent)

  // Track the most recently set camera so updateCameraAspectRatio can re-apply it
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var currentEye:    Vector[3] = initialPos
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var currentLookAt: Vector[3] = initialLookat
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var currentUp:     Vector[3] = initialUp

  def updateCamera(renderer: OptiXRenderer, eye: Vector[3], lookAt: Vector[3], up: Vector[3]): Unit =
    currentEye    = eye
    currentLookAt = lookAt
    currentUp     = up
    renderer.setCamera(eye, lookAt, up, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Updated camera: eye=(${eye(0)},${eye(1)},${eye(2)}), lookAt=(${lookAt(0)},${lookAt(1)},${lookAt(2)}), up=(${up(0)},${up(1)},${up(2)})")

  def updateCameraAspectRatio(renderer: OptiXRenderer, size: ImageSize): Unit =
    // Update cached image dimensions BEFORE re-applying the camera
    renderer.updateImageDimensions(size)
    renderer.setCamera(currentEye, currentLookAt, currentUp, horizontalFovDegrees = horizontalFov)
