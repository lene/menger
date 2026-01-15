package menger

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3
import menger.cli.Axis
import menger.cli.PlaneSpec
import menger.common.Const
import menger.config.CameraConfig
import menger.config.EnvironmentConfig
import menger.config.ExecutionConfig
import menger.config.MaterialConfig
import menger.config.OptiXEngineConfig
import menger.config.SceneConfig
import menger.engines.OptiXEngine
import menger.optix.RenderConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


class OptiXEngineSuite extends AnyFlatSpec with Matchers:

  private def createConfig(
    radius: Float = Const.defaultSphereRadius,
    ior: Float = Const.iorVacuum,
    color: Color = Color.WHITE,
    scale: Float = 1.0f,
    timeout: Float = 0f,
    enableStats: Boolean = false,
    renderConfig: RenderConfig = RenderConfig.Default
  ): OptiXEngineConfig =
    val colorHex = f"#${(color.r * 255).toInt}%02X${(color.g * 255).toInt}%02X${(color.b * 255).toInt}%02X"
    val objectSpec = ObjectSpec.parse(s"type=sphere:pos=0,0,0:size=${radius * 2}:scale=$scale:color=$colorHex:ior=$ior") match
      case Left(error) => sys.error(s"Failed to parse object spec: $error")
      case Right(spec) => List(spec)

    OptiXEngineConfig(
      scene = SceneConfig(objectSpecs = Some(objectSpec)),
      camera = CameraConfig(
        position = Vector3(0f, 0.5f, Const.defaultCameraZDistance),
        lookAt = Vector3(0f, 0f, 0f),
        up = Vector3(0f, 1f, 0f)
      ),
      environment = EnvironmentConfig(
        plane = PlaneSpec(Axis.Y, false, Const.defaultFloorPlaneY)
      ),
      execution = ExecutionConfig(
        fpsLogIntervalMs = 1000,
        timeout = timeout,
        enableStats = enableStats
      ),
      render = renderConfig
    )

  private def createEngine(config: OptiXEngineConfig): OptiXEngine =
    given ProfilingConfig = ProfilingConfig.disabled
    OptiXEngine(config)

  "OptiXEngine" should "be instantiated" in:
    val config = createConfig()
    val engine = createEngine(config)
    engine should not be null

  it should "store timeout in config" in:
    val config = createConfig(timeout = 5.0f)
    config.execution.timeout shouldBe 5.0f

  it should "accept various radius values" in:
    createConfig(radius = 0.1f)
    createConfig(radius = 10.0f)
    createConfig(radius = 1.5f)
    // No assertions - just verify parsing works

  it should "have default timeout 0" in:
    val config = createConfig()
    config.execution.timeout shouldBe 0f

  it should "store enableStats false by default" in:
    val config = createConfig()
    config.execution.enableStats shouldBe false

  it should "store enableStats true when provided" in:
    val config = createConfig(enableStats = true)
    config.execution.enableStats shouldBe true

  "OptiXEngineConfig" should "have sensible defaults" in:
    val config = OptiXEngineConfig.Default
    config.execution.timeout shouldBe 0f
    config.execution.enableStats shouldBe false
    config.render shouldBe RenderConfig.Default

  "MaterialConfig" should "have useful presets" in:
    MaterialConfig.Glass.ior shouldBe 1.5f
    MaterialConfig.Diamond.ior shouldBe 2.42f
    MaterialConfig.Water.ior shouldBe 1.33f
    MaterialConfig.Mirror.ior shouldBe 1.0f
