package menger.engines

import io.github.lene.optix.CameraState
import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.SceneConfigurator
import menger.common.Light
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.dsl.DenoiseMode
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BaseEngineDenoiseSuite extends AnyFlatSpec with Matchers with MockFactory:

  given ProfilingConfig = ProfilingConfig.disabled

  private val eye    = menger.common.Vector[3](0f, 0f, 3f)
  private val lookAt = menger.common.Vector[3](0f, 0f, 0f)
  private val up     = menger.common.Vector[3](0f, 1f, 0f)

  "configureOutputMode" should "enable denoising when denoiseMode is Final" in:
    val engine   = TestDenoiseEngine(denoiseMode = DenoiseMode.Final, accumulationFrames = 1)
    val renderer = mock[OptiXRenderer]
    (renderer.setDenoisingEnabled _).expects(true).once()
    engine.configureOutputMode(renderer)

  it should "disable denoising when denoiseMode is Off" in:
    val engine   = TestDenoiseEngine(denoiseMode = DenoiseMode.Off, accumulationFrames = 1)
    val renderer = mock[OptiXRenderer]
    (renderer.setDenoisingEnabled _).expects(false).once()
    engine.configureOutputMode(renderer)

  it should "call setAccumulationFrames only when frames > 1" in:
    val engine   = TestDenoiseEngine(denoiseMode = DenoiseMode.Off, accumulationFrames = 4)
    val renderer = mock[OptiXRenderer]
    (renderer.setDenoisingEnabled _).expects(false).once()
    (renderer.setAccumulationFrames _).expects(4).once()
    engine.configureOutputMode(renderer)

  it should "not call setAccumulationFrames when frames == 1" in:
    val engine   = TestDenoiseEngine(denoiseMode = DenoiseMode.Off, accumulationFrames = 1)
    val renderer = mock[OptiXRenderer]
    (renderer.setDenoisingEnabled _).expects(false).once()
    (renderer.setAccumulationFrames _).expects(*).never()
    engine.configureOutputMode(renderer)

  private final class TestDenoiseEngine(
    override protected val denoiseMode: DenoiseMode,
    override protected val accumulationFrames: Int
  ) extends BaseEngine(64):
    override protected def sceneConfigurator: SceneConfigurator =
      SceneConfigurator(eye, lookAt, up, Array.empty[Light])

    override protected def cameraState: CameraState =
      CameraState(eye, lookAt, up)

    override protected def renderConfig: RenderConfig =
      RenderConfig.Default

    override protected def textureDir: String =
      "."

    override def configureOutputMode(renderer: OptiXRenderer): Unit =
      super.configureOutputMode(renderer)
