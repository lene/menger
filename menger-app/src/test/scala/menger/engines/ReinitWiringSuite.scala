package menger.engines

import java.util.concurrent.atomic.AtomicInteger

import scala.language.implicitConversions

import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.TextureIndex
import menger.common.CausticsConfig
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.config.ExecutionConfig
import menger.config.TAnimationConfig
import menger.dsl.Camera
import menger.dsl.IBL
import menger.dsl.Scene
import menger.dsl.Sphere
import menger.optix.CameraState
import menger.optix.SceneConfigurator
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** menger#198: the animation and the preview call `ReinitRestore.restore` after a frame's
  * build, with the capacity the renderer had before it. A renderer whose capacity changes during
  * the build is the one a builder's `reinitialize` leaves behind. */
class ReinitWiringSuite extends AnyFlatSpec with Matchers with MockFactory:

  given ProfilingConfig = ProfilingConfig.disabled

  private val StartCapacity = 64
  private val GrownCapacity = 512
  private val sky = "assets/hdri/sky.hdr"
  private val skyTexture = TextureIndex(7)
  private val aScene = Scene(
    Camera.Default, List(Sphere(menger.dsl.Material.Glass)), envMap = Some(sky), ibl = Some(IBL())
  )
  private val caustics = CausticsConfig.Disabled
  private val configs = SceneConverter.convert(aScene, caustics, _ => None)

  private class AnimationUnderTest extends AnimationEngine(
    sceneFunction = _ => aScene,
    animConfig = TAnimationConfig(startT = 0f, endT = 1f, frames = 2, savePattern = "f-%d.png"),
    executionConfig = ExecutionConfig(),
    renderConfig = RenderConfig.Default,
    causticsConfig = caustics
  )

  private class PreviewUnderTest extends BaseEngine(StartCapacity) with WithPreview:
    override protected def sceneFunction: Float => Scene = _ => aScene
    override def previewConfig: TAnimationConfig =
      TAnimationConfig(startT = 0f, endT = 1f, frames = 2, savePattern = "f-%d.png")
    override protected def renderConfig: RenderConfig = RenderConfig.Default
    override protected def causticsConfig: CausticsConfig = caustics
    override protected def firstFrameConfigs: SceneConverter.SceneConfigs = configs
    override protected def textureDir: String = "."
    override protected val sceneConfigurator: SceneConfigurator =
      SceneConfigurator(configs.camera.position, configs.camera.lookAt, configs.camera.up, Array())
    override protected val cameraState: CameraState =
      CameraState(configs.camera.position, configs.camera.lookAt, configs.camera.up)

  // A renderer whose capacity is what the test says it is now.
  private def rendererWithCapacity(capacity: AtomicInteger): OptiXRenderer =
    val renderer = mock[OptiXRenderer]
    (() => renderer.maxInstances).expects().onCall(() => capacity.get).anyNumberOfTimes()
    renderer

  // The first frame sets the sky for the renderer (nothing was applied yet); only the second one
  // is the frame whose build grows the capacity, and sets the rest again.
  private def expectSkyTwiceAndSettingsOnce(renderer: OptiXRenderer): Unit =
    (renderer.uploadTextureFromFile _).expects(*).returning(skyTexture).twice()
    (renderer.setEnvironmentMap _).expects(skyTexture).twice()
    (renderer.setIBL _).expects(true, configs.iblStrength, configs.iblSamples).twice()
    (renderer.clearIBLSunExclusion _).expects().twice()
    (renderer.setRenderConfig _).expects(RenderConfig.Default).once()
    (renderer.setDenoisingEnabled _).expects(false).once()
    (renderer.setCausticsConfig _).expects(caustics).once()

  "WithAnimation.restoringAfter" should "set the sky and settings again after the renderer grew" in:
    val capacity = AtomicInteger(StartCapacity)
    val renderer = rendererWithCapacity(capacity)
    expectSkyTwiceAndSettingsOnce(renderer)
    val engine = AnimationUnderTest()

    engine.restoringAfter(renderer, configs, Nil, 0f)(())
    engine.restoringAfter(renderer, configs, Nil, 0f)(capacity.set(GrownCapacity))

  it should "return what the build returns" in:
    val capacity = AtomicInteger(StartCapacity)
    val renderer = rendererWithCapacity(capacity)
    (renderer.uploadTextureFromFile _).expects(*).returning(skyTexture).once()
    (renderer.setEnvironmentMap _).expects(*).once()
    (renderer.setIBL _).expects(*, *, *).once()
    (renderer.clearIBLSunExclusion _).expects().once()

    AnimationUnderTest().restoringAfter(renderer, configs, Nil, 0f)(42) shouldBe 42

  "WithPreview.restoringAfter" should "set the sky and settings again after the renderer grew" in:
    val capacity = AtomicInteger(StartCapacity)
    val renderer = rendererWithCapacity(capacity)
    expectSkyTwiceAndSettingsOnce(renderer)
    val engine = PreviewUnderTest()

    engine.restoringAfter(renderer, configs)(())
    engine.restoringAfter(renderer, configs)(capacity.set(GrownCapacity))

  it should "return what the build returns" in:
    val capacity = AtomicInteger(StartCapacity)
    val renderer = rendererWithCapacity(capacity)
    (renderer.uploadTextureFromFile _).expects(*).returning(skyTexture).once()
    (renderer.setEnvironmentMap _).expects(*).once()
    (renderer.setIBL _).expects(*, *, *).once()
    (renderer.clearIBLSunExclusion _).expects().once()

    PreviewUnderTest().restoringAfter(renderer, configs)(42) shouldBe 42

  // The video's texture index belongs to the renderer a reinitialize recreated.
  "WithAnimation.envMapVideoStateAfter" should "drop the state of a reinitialized renderer" in:
    WithAnimation.envMapVideoStateAfter(Some("state"), reinitialized = true) shouldBe None

  it should "keep the state of a renderer that was not reinitialized" in:
    WithAnimation.envMapVideoStateAfter(Some("state"), reinitialized = false) shouldBe Some("state")

  it should "have nothing to drop without a state" in:
    WithAnimation.envMapVideoStateAfter(None, reinitialized = true) shouldBe None
