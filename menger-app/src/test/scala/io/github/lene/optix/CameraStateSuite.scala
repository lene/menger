package io.github.lene.optix

import menger.common.ImageSize
import menger.common.Vector
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CameraStateSuite extends AnyFlatSpec with Matchers with MockFactory:

  private val pos    = Vector[3](0f, 0f, 5f)
  private val lookAt = Vector[3](0f, 0f, 0f)
  private val up     = Vector[3](0f, 1f, 0f)

  "updateCamera" should "apply the given camera on the renderer" in:
    val renderer = mock[OptiXRenderer]
    (renderer.setCamera _).expects(pos, lookAt, up, *).once()
    new CameraState(pos, lookAt, up).updateCamera(renderer, pos, lookAt, up)

  "updateCameraAspectRatio" should "update dimensions then re-apply the cached camera" in:
    val renderer = mock[OptiXRenderer]
    inSequence {
      (renderer.updateImageDimensions(_: ImageSize)).expects(ImageSize(800, 600)).once()
      (renderer.setCamera _).expects(pos, lookAt, up, *).once()
    }
    new CameraState(pos, lookAt, up).updateCameraAspectRatio(renderer, ImageSize(800, 600))

  it should "re-apply the most recently set camera, not the initial one" in:
    val renderer = mock[OptiXRenderer]
    val newEye = Vector[3](1f, 2f, 3f)
    inSequence {
      (renderer.setCamera _).expects(newEye, lookAt, up, *).once()
      (renderer.updateImageDimensions(_: ImageSize)).expects(*).once()
      (renderer.setCamera _).expects(newEye, lookAt, up, *).once()
    }
    val state = new CameraState(pos, lookAt, up)
    state.updateCamera(renderer, newEye, lookAt, up)
    state.updateCameraAspectRatio(renderer, ImageSize(640, 480))
