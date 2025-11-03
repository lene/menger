package menger

import com.badlogic.gdx.graphics.Color
import menger.engines.OptiXEngine
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Tests for OptiXEngine.
 *
 * Note: Full lifecycle tests require LibGDX application context and OptiX runtime.
 * These tests verify code structure, parameters, and basic logic without full rendering.
 */
class OptiXEngineTest extends AnyFlatSpec with Matchers:

  // Dummy rotation/projection parameters for testing
  private def dummyRotProj = RotationProjectionParameters(
    rotXW = 0f, rotYW = 0f, rotZW = 0f,
    eyeW = 3f, screenW = 2f,
    rotX = 0f, rotY = 0f, rotZ = 0f
  )

  private def createEngine(radius: Float = 1.5f, ior: Float = 1.0f, scale: Float = 1.0f, timeout: Float = 0f): OptiXEngine =
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
      ior = ior,
      scale = scale,
      timeout = timeout
    )

  "OptiXEngine" should "be instantiated" in:
    val engine = createEngine()
    engine should not be null

  it should "store sphere radius" in:
    val engine = createEngine(radius = 2.5f)
    engine.sphereRadius shouldBe 2.5f

  it should "store timeout" in:
    val engine = createEngine(timeout = 5.0f)
    engine.timeout shouldBe 5.0f

  it should "store color" in:
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
      ior = 1.0f,
      scale = 1.0f,
      timeout = 0f
    )
    engine.color shouldBe Color.RED

  it should "store color with transparency" in:
    given ProfilingConfig = ProfilingConfig.disabled
    val semiTransparentGreen = new Color(0f, 1f, 0.5f, 0.5f)  // Green with 50% alpha
    val engine = new OptiXEngine(
      spongeType = "sphere",
      spongeLevel = 0f,
      rotationProjectionParameters = dummyRotProj,
      lines = false,
      color = semiTransparentGreen,
      faceColor = None,
      lineColor = None,
      fpsLogIntervalMs = 1000,
      sphereRadius = 1.0f,
      ior = 1.0f,
      scale = 1.0f,
      timeout = 0f
    )
    engine.color.r shouldBe 0f +- 0.01f
    engine.color.g shouldBe 1f +- 0.01f
    engine.color.b shouldBe 0.5f +- 0.01f
    engine.color.a shouldBe 0.5f +- 0.01f

  it should "store IOR" in:
    val engine = createEngine(ior = 1.5f)
    engine.ior shouldBe 1.5f

  it should "have default IOR 1.0" in:
    val engine = createEngine()
    engine.ior shouldBe 1.0f

  it should "have sponge type sphere" in:
    val engine = createEngine()
    engine.spongeType shouldBe "sphere"

  it should "accept various radius values" in:
    val engine1 = createEngine(radius = 0.1f)
    engine1.sphereRadius shouldBe 0.1f

    val engine2 = createEngine(radius = 10.0f)
    engine2.sphereRadius shouldBe 10.0f

    val engine3 = createEngine(radius = 1.5f)
    engine3.sphereRadius shouldBe 1.5f

  it should "have default timeout 0" in:
    val engine = createEngine()
    engine.timeout shouldBe 0f
