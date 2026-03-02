package menger.optix

import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.Vector3Extensions.toVector3
import menger.common.ImageSize

class CameraState(
  initialPos: Vector3,
  initialLookat: Vector3,
  initialUp: Vector3
) extends LazyLogging:

  private val horizontalFov = 45f  // Fixed horizontal FOV in degrees (aspect-ratio independent)

  // Track the most recently set camera so updateCameraAspectRatio can re-apply it
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var currentEye:    Vector3 = initialPos
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var currentLookAt: Vector3 = initialLookat
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var currentUp:     Vector3 = initialUp

  def updateCamera(renderer: OptiXRenderer, eye: Vector3, lookAt: Vector3, up: Vector3): Unit =
    currentEye    = eye
    currentLookAt = lookAt
    currentUp     = up
    val eyeVec    = eye.toVector3
    val lookAtVec = lookAt.toVector3
    val upVec     = up.toVector3
    renderer.setCamera(eyeVec, lookAtVec, upVec, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Updated camera: eye=(${eyeVec(0)},${eyeVec(1)},${eyeVec(2)}), lookAt=(${lookAtVec(0)},${lookAtVec(1)},${lookAtVec(2)}), up=(${upVec(0)},${upVec(1)},${upVec(2)})")

  def updateCameraAspectRatio(renderer: OptiXRenderer, size: ImageSize): Unit =
    // Update cached image dimensions BEFORE re-applying the camera
    renderer.updateImageDimensions(size)
    val eyeVec    = currentEye.toVector3
    val lookAtVec = currentLookAt.toVector3
    val upVec     = currentUp.toVector3
    renderer.setCamera(eyeVec, lookAtVec, upVec, horizontalFovDegrees = horizontalFov)
