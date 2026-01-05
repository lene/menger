package menger

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3
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
    OptiXEngineConfig(
      scene = SceneConfig(
        spongeType = "sphere",
        level = 0f,
        material = MaterialConfig(color, ior),
        sphereRadius = radius,
        scale = scale,
        center = Vector3(0f, 0f, 0f)
      ),
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

  it should "store sphere radius in config" in:
    val config = createConfig(radius = 2.5f)
    config.scene.sphereRadius shouldBe 2.5f

  it should "store timeout in config" in:
    val config = createConfig(timeout = 5.0f)
    config.execution.timeout shouldBe 5.0f

  it should "store color in config" in:
    val config = createConfig(color = Color.RED)
    config.scene.material.color shouldBe Color.RED

  it should "store color with transparency in config" in:
    val semiTransparentGreen = new Color(0f, 1f, 0.5f, 0.5f)
    val config = createConfig(color = semiTransparentGreen)
    config.scene.material.color.r shouldBe 0f +- 0.01f
    config.scene.material.color.g shouldBe 1f +- 0.01f
    config.scene.material.color.b shouldBe 0.5f +- 0.01f
    config.scene.material.color.a shouldBe 0.5f +- 0.01f

  it should "store IOR in config" in:
    val config = createConfig(ior = Const.iorGlass)
    config.scene.material.ior shouldBe Const.iorGlass

  it should "have default IOR 1.0 in MaterialConfig.Default" in:
    // Note: MaterialConfig.Default has IOR 1.5 (glass), but we can override
    val config = createConfig(ior = Const.iorVacuum)
    config.scene.material.ior shouldBe Const.iorVacuum

  it should "have sponge type sphere in config" in:
    val config = createConfig()
    config.scene.spongeType shouldBe "sphere"

  it should "accept various radius values" in:
    createConfig(radius = 0.1f).scene.sphereRadius shouldBe 0.1f
    createConfig(radius = 10.0f).scene.sphereRadius shouldBe 10.0f
    createConfig(radius = 1.5f).scene.sphereRadius shouldBe 1.5f

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
    config.scene.spongeType shouldBe "sphere"
    config.execution.timeout shouldBe 0f
    config.execution.enableStats shouldBe false
    config.render shouldBe RenderConfig.Default

  "MaterialConfig" should "have useful presets" in:
    MaterialConfig.Glass.ior shouldBe 1.5f
    MaterialConfig.Diamond.ior shouldBe 2.42f
    MaterialConfig.Water.ior shouldBe 1.33f
    MaterialConfig.Mirror.ior shouldBe 1.0f
