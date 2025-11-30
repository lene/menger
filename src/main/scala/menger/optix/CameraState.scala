package menger.optix

import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.common.ImageSize
import menger.common.Vector

class CameraState(
  initialPos: Vector3,
  initialLookat: Vector3,
  initialUp: Vector3
) extends LazyLogging:

  private val horizontalFov = 45f  // Fixed horizontal FOV in degrees (aspect-ratio independent)

  def updateCamera(renderer: OptiXRenderer, eye: Vector3, lookAt: Vector3, up: Vector3): Unit =
    val eyeVec = Vector[3](eye.x, eye.y, eye.z)
    val lookAtVec = Vector[3](lookAt.x, lookAt.y, lookAt.z)
    val upVec = Vector[3](up.x, up.y, up.z)
    renderer.setCamera(eyeVec, lookAtVec, upVec, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Updated camera: eye=(${eyeVec(0)},${eyeVec(1)},${eyeVec(2)}), lookAt=(${lookAtVec(0)},${lookAtVec(1)},${lookAtVec(2)}), up=(${upVec(0)},${upVec(1)},${upVec(2)})")

  def updateCameraAspectRatio(renderer: OptiXRenderer, size: ImageSize): Unit =
    // Update cached image dimensions BEFORE calling setCamera
    renderer.updateImageDimensions(size)

    val eye = Vector[3](initialPos.x, initialPos.y, initialPos.z)
    val lookAt = Vector[3](initialLookat.x, initialLookat.y, initialLookat.z)
    val up = Vector[3](initialUp.x, initialUp.y, initialUp.z)

    renderer.setCamera(eye, lookAt, up, horizontalFovDegrees = horizontalFov)
