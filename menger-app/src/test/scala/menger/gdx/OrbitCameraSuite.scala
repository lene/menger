package menger.gdx

import com.badlogic.gdx.math.Vector3
import menger.common.MouseButton
import menger.common.ScreenCoords
import menger.input.OrbitConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class OrbitCameraSuite extends AnyFlatSpec with Matchers:

  private val defaultEye    = Vector3(0f, 0.5f, 3f)
  private val defaultLookAt = Vector3(0f, 0f, 0f)
  private val defaultUp     = Vector3(0f, 1f, 0f)

  private def makeCamera = OrbitCamera(defaultEye, defaultLookAt, defaultUp)

  "OrbitCamera.currentEye" should "return a copy of the initial eye" in {
    val cam = makeCamera
    val eye = cam.currentEye
    eye.x shouldBe (defaultEye.x +- 0.001f)
    eye.y shouldBe (defaultEye.y +- 0.001f)
    eye.z shouldBe (defaultEye.z +- 0.001f)
  }

  it should "return a defensive copy (mutation does not affect camera)" in {
    val cam = makeCamera
    val eye = cam.currentEye
    eye.x = 999f
    cam.currentEye.x should not be 999f
  }

  "OrbitCamera.currentLookAt" should "return a copy of the initial lookAt" in {
    val cam    = makeCamera
    val lookAt = cam.currentLookAt
    lookAt.x shouldBe (defaultLookAt.x +- 0.001f)
    lookAt.y shouldBe (defaultLookAt.y +- 0.001f)
    lookAt.z shouldBe (defaultLookAt.z +- 0.001f)
  }

  "OrbitCamera.currentUp" should "return a copy of the initial up" in {
    val cam = makeCamera
    val up  = cam.currentUp
    up.x shouldBe (defaultUp.x +- 0.001f)
    up.y shouldBe (defaultUp.y +- 0.001f)
    up.z shouldBe (defaultUp.z +- 0.001f)
  }

  "OrbitCamera.orbit" should "change eye position" in {
    val cam  = makeCamera
    val eye0 = cam.currentEye
    cam.orbit(50, 0)
    val eye1 = cam.currentEye
    // Horizontal orbit changes x/z but not y significantly
    (eye1.x - eye0.x).abs + (eye1.z - eye0.z).abs should be > 0.01f
  }

  it should "maintain eye-to-lookAt distance after orbit" in {
    val cam  = makeCamera
    val dist0 = cam.currentEye.dst(cam.currentLookAt)
    cam.orbit(30, 15)
    val dist1 = cam.currentEye.dst(cam.currentLookAt)
    dist1 shouldBe (dist0 +- 0.01f)
  }

  "OrbitCamera.pan" should "move both eye and lookAt by the same offset" in {
    val cam     = makeCamera
    val eye0    = cam.currentEye.cpy()
    val lookAt0 = cam.currentLookAt.cpy()
    cam.pan(100, 0)
    val eyeDelta    = cam.currentEye.cpy().sub(eye0)
    val lookAtDelta = cam.currentLookAt.cpy().sub(lookAt0)
    eyeDelta.x shouldBe (lookAtDelta.x +- 0.001f)
    eyeDelta.y shouldBe (lookAtDelta.y +- 0.001f)
    eyeDelta.z shouldBe (lookAtDelta.z +- 0.001f)
  }

  it should "preserve eye-to-lookAt distance" in {
    val cam   = makeCamera
    val dist0 = cam.currentEye.dst(cam.currentLookAt)
    cam.pan(100, 50)
    val dist1 = cam.currentEye.dst(cam.currentLookAt)
    dist1 shouldBe (dist0 +- 0.01f)
  }

  "OrbitCamera.zoom" should "change eye-to-lookAt distance" in {
    val cam   = makeCamera
    val dist0 = cam.currentEye.dst(cam.currentLookAt)
    cam.zoom(-1f)  // zoom in
    val dist1 = cam.currentEye.dst(cam.currentLookAt)
    dist1 should be < dist0
  }

  it should "not move lookAt" in {
    val cam     = makeCamera
    val lookAt0 = cam.currentLookAt.cpy()
    cam.zoom(1f)
    cam.currentLookAt.x shouldBe (lookAt0.x +- 0.001f)
    cam.currentLookAt.y shouldBe (lookAt0.y +- 0.001f)
    cam.currentLookAt.z shouldBe (lookAt0.z +- 0.001f)
  }

  "OrbitCamera.moveDrag" should "return None when no drag active" in {
    val cam = makeCamera
    cam.moveDrag(ScreenCoords(10, 20)) shouldBe None
  }

  it should "return delta after startDrag" in {
    val cam = makeCamera
    cam.startDrag(ScreenCoords(10, 20), MouseButton.Left)
    val result = cam.moveDrag(ScreenCoords(15, 25))
    result shouldBe Some((5, 5, MouseButton.Left))
  }

  it should "accumulate deltas across multiple moveDrag calls" in {
    val cam = makeCamera
    cam.startDrag(ScreenCoords(0, 0), MouseButton.Left)
    cam.moveDrag(ScreenCoords(10, 0))
    val result = cam.moveDrag(ScreenCoords(15, 0))
    result shouldBe Some((5, 0, MouseButton.Left))
  }

  it should "return None after endDrag" in {
    val cam = makeCamera
    cam.startDrag(ScreenCoords(0, 0), MouseButton.Left)
    cam.endDrag()
    cam.moveDrag(ScreenCoords(10, 20)) shouldBe None
  }

  it should "preserve button from startDrag" in {
    val cam = makeCamera
    cam.startDrag(ScreenCoords(0, 0), MouseButton.Right)
    val result = cam.moveDrag(ScreenCoords(5, 5))
    result.map(_._3) shouldBe Some(MouseButton.Right)
  }

  "OrbitCamera" should "accept custom OrbitConfig" in {
    val cam = OrbitCamera(defaultEye, defaultLookAt, defaultUp, OrbitConfig(maxDistance = 100f))
    // Just verify construction doesn't throw
    cam.currentEye.z shouldBe (defaultEye.z +- 0.001f)
  }
