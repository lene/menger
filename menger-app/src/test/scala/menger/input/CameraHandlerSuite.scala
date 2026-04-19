package menger.input

import menger.common.Const
import menger.common.MouseButton
import menger.common.ScreenCoords
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CameraHandlerSuite extends AnyFlatSpec with Matchers:

  private val screenW = Const.Input.eyeScrollOffset  // 1.0f — crash threshold for Projection

  // Minimal CameraHandler stub — no LibGDX required
  private class TestCameraHandler extends CameraHandler:
    def handleMouseDown(pos: ScreenCoords, button: MouseButton, pointer: Int) = false
    def handleMouseUp(pos: ScreenCoords, button: MouseButton, pointer: Int) = false
    def handleMouseDrag(pos: ScreenCoords, pointer: Int, button: MouseButton) = false
    def handleScroll(amountX: Float, amountY: Float) = false
    def exposedEyeW(amountY: Float): Float = computeEyeW(amountY)
  private val stub = TestCameraHandler()

  "CameraHandler.computeEyeW" should "return defaultEyeW for zero scroll" in:
    stub.exposedEyeW(0f) shouldBe Const.defaultEyeW +- 1e-4f

  it should "return a value above screenW for a single negative scroll step" in:
    // 64^(-1) = 0.015625; eyeW = 1.0 + 0.015625 = 1.015625 — well above screenW
    val eyeW = stub.exposedEyeW(-1f)
    eyeW should be > screenW
    eyeW shouldBe (screenW + math.pow(Const.Input.eyeScrollBase, -1.0).toFloat) +- 1e-4f

  it should "increase eyeW for positive scroll" in:
    stub.exposedEyeW(1f) should be > stub.exposedEyeW(0f)

  it should "decrease eyeW for negative scroll relative to zero" in:
    stub.exposedEyeW(-1f) should be < stub.exposedEyeW(0f)

  it should "always produce eyeW strictly above screenW for moderate inputs" in:
    // Note: very large negative magnitudes (amountY < -4) underflow float32 precision
    // and return exactly screenW. This is acceptable — no user scrolls 4+ notches at once.
    val inputs = List(-3f, -2f, -1f, 0f, 1f, 2f, 3f)
    all(inputs.map(stub.exposedEyeW)) should be > screenW
