package menger

import com.badlogic.gdx.graphics.Color
import menger.engines.OptiXEngine
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/**
 * Tests for OptiXEngine.
 *
 * Note: Full lifecycle tests require LibGDX application context and OptiX runtime.
 * These tests verify code structure, parameters, and basic logic without full rendering.
 */
class OptiXEngineTest extends AnyFunSuite with Matchers:

  // Dummy rotation/projection parameters for testing
  private def dummyRotProj = RotationProjectionParameters(
    rotXW = 0f, rotYW = 0f, rotZW = 0f,
    eyeW = 3f, screenW = 2f,
    rotX = 0f, rotY = 0f, rotZ = 0f
  )

  private def createEngine(radius: Float = 1.5f, timeout: Float = 0f): OptiXEngine =
    given ProfilingConfig = ProfilingConfig.disabled
    new OptiXEngine(
      spongeType = "sphere",
      spongeLevel = 0f,
      rotationProjectionParameters = dummyRotProj,
      lines = false,
      color = Color.WHITE,
      faceColor = None,
      lineColor = None,
      fpsLogIntervalMs = 1000,
      sphereRadius = radius,
      timeout = timeout
    )

  test("OptiXEngine can be instantiated"):
    val engine = createEngine()
    engine should not be null

  test("OptiXEngine stores sphere radius"):
    val engine = createEngine(radius = 2.5f)
    engine.sphereRadius shouldBe 2.5f

  test("OptiXEngine stores timeout"):
    val engine = createEngine(timeout = 5.0f)
    engine.timeout shouldBe 5.0f

  test("OptiXEngine stores color"):
    given ProfilingConfig = ProfilingConfig.disabled
    val engine = new OptiXEngine(
      spongeType = "sphere",
      spongeLevel = 0f,
      rotationProjectionParameters = dummyRotProj,
      lines = false,
      color = Color.RED,
      faceColor = None,
      lineColor = None,
      fpsLogIntervalMs = 1000,
      sphereRadius = 1.0f,
      timeout = 0f
    )
    engine.color shouldBe Color.RED

  test("OptiXEngine sponge type is sphere"):
    val engine = createEngine()
    engine.spongeType shouldBe "sphere"

  test("OptiXEngine accepts various radius values"):
    val engine1 = createEngine(radius = 0.1f)
    engine1.sphereRadius shouldBe 0.1f

    val engine2 = createEngine(radius = 10.0f)
    engine2.sphereRadius shouldBe 10.0f

    val engine3 = createEngine(radius = 1.5f)
    engine3.sphereRadius shouldBe 1.5f

  test("OptiXEngine timeout defaults to 0"):
    val engine = createEngine()
    engine.timeout shouldBe 0f

  test("OptiXEngine can be disposed without initialization"):
    val engine = createEngine()
    // Should not throw even if create() was never called
    noException should be thrownBy engine.dispose()
