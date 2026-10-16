package menger.engines

import io.github.lene.optix.CameraState
import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.SceneConfigurator
import menger.ObjectSpec
import menger.common.CausticsConfig
import menger.common.Light
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.common.Vector
import menger.config.CameraConfig
import menger.config.SceneConfig
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// Review 2026-10-14 (menger#101): a reload is checked before the current scene is torn down,
// and a scene over the budget never reaches the renderer on any build path. The mocked renderer
// has no expectations, so any call on it fails the test.
class LiveReloadSuite extends AnyFlatSpec with Matchers with MockFactory:

  given ProfilingConfig = ProfilingConfig.disabled

  private val volume5 = ObjectSpec(objectType = "sponge-volume", level = Some(5f))

  "LiveReload.precheck" should "accept a scene the renderer can build" in:
    TestEngine().check(List(ObjectSpec("sphere"), ObjectSpec("cube"))).isSuccess shouldBe true

  it should "refuse a scene over the scene budget" in:
    val result = TestEngine().check(List(volume5, volume5))
    result.failed.get.getMessage should include("scene over budget")

  // The F94 scene: edges of a level-3 tesseract sponge need more instances than the renderer has.
  it should "refuse a scene its builder rejects, with the validator's own reason" in:
    val edges =
      ObjectSpec(objectType = "tesseract-sponge", level = Some(3f), edgeRadius = Some(0.02f))
    TestEngine().check(List(edges)).failed.get.getMessage should include("instances")

  "BaseEngine.buildSceneFromConfigs" should "refuse an over-budget scene before the renderer" in:
    val renderer = mock[OptiXRenderer]
    val result = TestEngine().build(configsFor(List(volume5, volume5)), renderer)
    result.failed.get.getMessage should include("scene over budget")

  "BuildPrecheck" should "find nothing in an empty scene" in:
    BuildPrecheck.problems(Nil) shouldBe empty

  private def configsFor(specs: List[ObjectSpec]): SceneConverter.SceneConfigs =
    SceneConverter.SceneConfigs(
      scene = SceneConfig.multiObject(specs),
      camera = CameraConfig.Default,
      lights = List.empty,
      caustics = CausticsConfig()
    )

  private final class TestEngine extends BaseEngine(64) with LiveReload:
    private val eye = Vector[3](0f, 0f, 3f)
    override protected def sceneConfigurator: SceneConfigurator =
      SceneConfigurator(eye, Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f), Array.empty[Light])
    override protected def cameraState: CameraState =
      CameraState(eye, Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f))
    override protected def renderConfig: RenderConfig = RenderConfig.Default
    override protected def textureDir: String = "."

    def check(specs: List[ObjectSpec]): scala.util.Try[Unit] = precheck(specs)
    def build(configs: SceneConverter.SceneConfigs, renderer: OptiXRenderer): scala.util.Try[Unit] =
      buildSceneFromConfigs(configs, renderer)
