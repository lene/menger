package menger.engines

import io.github.lene.optix.CameraState
import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.SceneConfigurator
import menger.ObjectSpec
import menger.common.CausticsConfig
import menger.common.Light
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.common.ValidationException
import menger.common.Vector
import menger.config.CameraConfig
import menger.config.SceneConfig
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BaseEngineSceneValidationSuite extends AnyFlatSpec with Matchers with MockFactory:

  given ProfilingConfig = ProfilingConfig.disabled

  private val eye    = Vector[3](0f, 0f, 3f)
  private val lookAt = Vector[3](0f, 0f, 0f)
  private val up     = Vector[3](0f, 1f, 0f)

  "buildSceneFromConfigs" should "validate triangle mesh configs before building" in:
    val engine   = TestEngine(maxInstances = 64)
    val renderer = mock[OptiXRenderer]
    val invalidSpec =
      ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get.copy(level = None)

    val result = engine.build(configsFor(List(invalidSpec)), renderer)

    val error = result.failed.get
    error shouldBe a[ValidationException]
    error.getMessage should include("Incompatible")

  it should "validate mixed config groups before building" in:
    val engine   = TestEngine(maxInstances = 1)
    val renderer = mock[OptiXRenderer]
    val specs    = List(ObjectSpec("sphere"), ObjectSpec("sphere"), ObjectSpec("cube"))

    val result = engine.build(configsFor(specs), renderer)

    val error = result.failed.get
    error shouldBe a[ValidationException]
    error.getMessage should include("Too many objects")

  private def configsFor(specs: List[ObjectSpec]): SceneConverter.SceneConfigs =
    SceneConverter.SceneConfigs(
      scene = SceneConfig.multiObject(specs),
      camera = CameraConfig.Default,
      lights = List.empty,
      caustics = CausticsConfig()
    )

  private final class TestEngine(maxInstances: Int) extends BaseEngine(maxInstances):
    override protected def sceneConfigurator: SceneConfigurator =
      SceneConfigurator(eye, lookAt, up, Array.empty[Light])

    override protected def cameraState: CameraState =
      CameraState(eye, lookAt, up)

    override protected def renderConfig: RenderConfig =
      RenderConfig.Default

    override protected def textureDir: String =
      "."

    def build(
      configs: SceneConverter.SceneConfigs,
      renderer: OptiXRenderer
    ): scala.util.Try[Unit] =
      buildSceneFromConfigs(configs, renderer)
