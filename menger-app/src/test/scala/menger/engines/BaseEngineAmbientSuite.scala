package menger.engines

import io.github.lene.optix.AmbientModel
import io.github.lene.optix.OptiXRenderer
import menger.common.Light
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.dsl.Ambient
import menger.optix.CameraState
import menger.optix.SceneConfigurator
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BaseEngineAmbientSuite extends AnyFlatSpec with Matchers with MockFactory:

  given ProfilingConfig = ProfilingConfig.disabled

  private val eye    = menger.common.Vector[3](0f, 0f, 3f)
  private val lookAt = menger.common.Vector[3](0f, 0f, 0f)
  private val up     = menger.common.Vector[3](0f, 1f, 0f)

  "AmbientMapping.toModel" should "map Constant to the constant model" in:
    AmbientMapping.toModel(Ambient.Constant) shouldBe AmbientModel.Constant

  it should "map Occlusion's samples and radius to the occlusion model" in:
    AmbientMapping.toModel(Ambient.Occlusion(64, 0.5f)) shouldBe AmbientModel.Occlusion(64, 0.5f)
    AmbientMapping.toModel(Ambient.Occlusion()) shouldBe AmbientModel.Occlusion()

  "configureOutputMode" should "set the ambient model the engine names" in:
    val engine   = TestAmbientEngine(Some(Ambient.Occlusion(32, 1f)))
    val renderer = mock[OptiXRenderer]
    (renderer.setDenoisingEnabled _).expects(false).once()
    (renderer.setAmbient _).expects(AmbientModel.Occlusion(32, 1f)).once()
    engine.configureOutputMode(renderer)

  it should "set the constant model" in:
    val engine   = TestAmbientEngine(Some(Ambient.Constant))
    val renderer = mock[OptiXRenderer]
    (renderer.setDenoisingEnabled _).expects(false).once()
    (renderer.setAmbient _).expects(AmbientModel.Constant).once()
    engine.configureOutputMode(renderer)

  it should "leave the renderer's own ambient default alone when none is named" in:
    val engine   = TestAmbientEngine(None)
    val renderer = mock[OptiXRenderer]
    (renderer.setDenoisingEnabled _).expects(false).once()
    (renderer.setAmbient _).expects(*).never()
    engine.configureOutputMode(renderer)

  private final class TestAmbientEngine(
    override protected val ambient: Option[Ambient]
  ) extends BaseEngine(64):
    override protected def sceneConfigurator: SceneConfigurator =
      SceneConfigurator(eye, lookAt, up, Array.empty[Light])

    override protected def cameraState: CameraState = CameraState(eye, lookAt, up)

    override protected def renderConfig: RenderConfig = RenderConfig.Default

    override protected def textureDir: String = "."

    override def configureOutputMode(renderer: OptiXRenderer): Unit =
      super.configureOutputMode(renderer)
