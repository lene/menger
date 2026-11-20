package menger.engines

import scala.language.implicitConversions

import io.github.lene.optix.AmbientModel
import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.TextureIndex
import menger.common.CausticsConfig
import menger.common.RenderConfig
import menger.config.IblSunExclusion
import menger.dsl.Ambient
import menger.dsl.Camera
import menger.dsl.DenoiseMode
import menger.dsl.IBL
import menger.dsl.Scene
import menger.dsl.Sphere
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** menger#198: a builder's `renderer.reinitialize` recreates the native renderer and drops the
  * env map, the image-based lighting, the caustics and the render settings; an animation frame
  * that grew the capacity must set them again after its build. */
class ReinitRestoreSuite extends AnyFlatSpec with Matchers with MockFactory:

  private val sky = "assets/hdri/sky.hdr"
  private val skyTexture = TextureIndex(7)
  private val videoTexture = TextureIndex(9)
  private val StartCapacity = 64
  private val GrownCapacity = 512

  private val caustics = CausticsConfig(enabled = true)
  private val settings = ReinitRestore.FrameSettings(
    RenderConfig(), DenoiseMode.Final, accumulationFrames = 4, Some(Ambient.Constant)
  )

  private def configs(envMap: Option[String] = Some(sky), ibl: Option[IBL] = Some(IBL())) =
    SceneConverter.convert(
      Scene(Camera.Default, List(Sphere(menger.dsl.Material.Glass)), envMap = envMap, ibl = ibl),
      caustics, _ => None
    )

  private def appliedAt(capacity: Int) =
    Some(IblSetup.Sky(Some(sky), None, capacity))

  private def rendererWithCapacity(capacity: Int): OptiXRenderer =
    val renderer = mock[OptiXRenderer]
    (() => renderer.maxInstances).expects().returning(capacity).anyNumberOfTimes()
    renderer

  "ReinitRestore.restore" should "set what a reinitialize drops, in the required order" in:
    val scene = configs()
    val renderer = rendererWithCapacity(GrownCapacity)
    inSequence:
      (renderer.setRenderConfig _).expects(settings.render).once()
      (renderer.setDenoisingEnabled _).expects(true).once()
      (renderer.setAccumulationFrames _).expects(4).once()
      (renderer.setAmbient _).expects(AmbientModel.Constant).once()
      (renderer.setCausticsConfig _).expects(caustics).once()
      // the static map first: the env-map video, set next, replaces it as the background
      (renderer.uploadTextureFromFile _).expects(where((p: String) => p.endsWith(sky)))
        .returning(skyTexture).once()
      (renderer.setEnvironmentMap _).expects(skyTexture).once()
      (renderer.setEnvironmentMap _).expects(videoTexture).once()
      // the lighting last: it samples the map the renderer holds
      (renderer.setIBL _).expects(true, scene.iblStrength, scene.iblSamples).once()
      (renderer.clearIBLSunExclusion _).expects().once()

    val after = ReinitRestore.restore(
      renderer, ".", scene, settings, StartCapacity, appliedAt(StartCapacity)
    )(_ => renderer.setEnvironmentMap(videoTexture))

    after shouldBe appliedAt(GrownCapacity)

  // The video's texture index belongs to the renderer a reinitialize recreated: its state has
  // to be dropped so the video is uploaded again.
  it should "tell the video step that the renderer was reinitialized" in:
    val renderer = rendererWithCapacity(GrownCapacity)
    val told = new java.util.concurrent.atomic.AtomicReference[Option[Boolean]](None)
    (renderer.uploadTextureFromFile _).expects(*).returning(skyTexture).once()
    (renderer.setEnvironmentMap _).expects(*).once()
    (renderer.setRenderConfig _).expects(*).once()
    (renderer.setDenoisingEnabled _).expects(*).once()
    (renderer.setAccumulationFrames _).expects(*).once()
    (renderer.setAmbient _).expects(*).once()
    (renderer.setCausticsConfig _).expects(*).once()
    (renderer.setIBL _).expects(*, *, *).once()
    (renderer.clearIBLSunExclusion _).expects().once()

    ReinitRestore.restore(
      renderer, ".", configs(), settings, StartCapacity, appliedAt(StartCapacity)
    )(reinitialized => told.set(Some(reinitialized)))

    told.get shouldBe Some(true)

  it should "tell the video step that the renderer was not reinitialized" in:
    val renderer = rendererWithCapacity(StartCapacity)
    val told = new java.util.concurrent.atomic.AtomicReference[Option[Boolean]](None)

    ReinitRestore.restore(
      renderer, ".", configs(), settings, StartCapacity, appliedAt(StartCapacity)
    )(reinitialized => told.set(Some(reinitialized)))

    told.get shouldBe Some(false)

  it should "set the sun exclusion of the scene again" in:
    val cone = IblSunExclusion(menger.common.Vector[3](0f, 0.6f, 0.8f), 2f)
    val scene = configs().copy(iblSunExclusion = Some(cone))
    val renderer = rendererWithCapacity(GrownCapacity)
    (renderer.setRenderConfig _).expects(*).once()
    (renderer.setDenoisingEnabled _).expects(*).once()
    (renderer.setAccumulationFrames _).expects(*).once()
    (renderer.setAmbient _).expects(*).once()
    (renderer.setCausticsConfig _).expects(*).once()
    (renderer.uploadTextureFromFile _).expects(*).returning(skyTexture).once()
    (renderer.setEnvironmentMap _).expects(skyTexture).once()
    (renderer.setIBL _).expects(*, *, *).once()
    (renderer.setIBLSunExclusion _).expects(cone.directionToSun, cone.halfAngleDegrees).once()

    ReinitRestore.restore(
      renderer, ".", scene, settings, StartCapacity, appliedAt(StartCapacity)
    )(_ => ())

  it should "touch nothing but the video when the renderer was not reinitialized" in:
    val renderer = rendererWithCapacity(StartCapacity)
    (renderer.setRenderConfig _).expects(*).never()
    (renderer.setDenoisingEnabled _).expects(*).never()
    (renderer.setAccumulationFrames _).expects(*).never()
    (renderer.setAmbient _).expects(*).never()
    (renderer.setCausticsConfig _).expects(*).never()
    (renderer.uploadTextureFromFile _).expects(*).never()
    (renderer.setIBL _).expects(*, *, *).never()
    (renderer.setEnvironmentMap _).expects(videoTexture).once()

    val after = ReinitRestore.restore(
      renderer, ".", configs(), settings, StartCapacity, appliedAt(StartCapacity)
    )(_ => renderer.setEnvironmentMap(videoTexture))

    after shouldBe appliedAt(StartCapacity)

  it should "read the map again, and set the lighting, when the scene's map changed" in:
    val renderer = rendererWithCapacity(StartCapacity)
    val scene = configs(envMap = Some("assets/hdri/other.hdr"))
    (renderer.setRenderConfig _).expects(*).never()
    (renderer.uploadTextureFromFile _).expects(where((p: String) => p.endsWith("other.hdr")))
      .returning(skyTexture).once()
    (renderer.setEnvironmentMap _).expects(skyTexture).once()
    (renderer.setIBL _).expects(true, scene.iblStrength, scene.iblSamples).once()
    (renderer.clearIBLSunExclusion _).expects().once()

    ReinitRestore.restore(
      renderer, ".", scene, settings, StartCapacity, appliedAt(StartCapacity)
    )(_ => ())

  it should "not read the map again in a frame after the reinitialize" in:
    val renderer = rendererWithCapacity(GrownCapacity)
    (renderer.uploadTextureFromFile _).expects(*).never()
    (renderer.setIBL _).expects(*, *, *).never()

    val after = ReinitRestore.restore(
      renderer, ".", configs(), settings, GrownCapacity, appliedAt(GrownCapacity)
    )(_ => ())

    after shouldBe appliedAt(GrownCapacity)

  it should "leave a scene without a map or lighting alone, whatever the capacity does" in:
    val renderer = rendererWithCapacity(GrownCapacity)
    val bare = configs(envMap = None, ibl = None)
    (renderer.setRenderConfig _).expects(*).once()
    (renderer.setDenoisingEnabled _).expects(*).once()
    (renderer.setAccumulationFrames _).expects(*).once()
    (renderer.setAmbient _).expects(*).once()
    (renderer.setCausticsConfig _).expects(*).once()
    (renderer.uploadTextureFromFile _).expects(*).never()
    (renderer.setEnvironmentMap _).expects(*).never()
    (renderer.setIBL _).expects(*, *, *).never()

    ReinitRestore.restore(
      renderer, ".", bare, settings, StartCapacity, Some(IblSetup.Sky(None, None, StartCapacity))
    )(_ => ())

  "ReinitRestore.FrameSettings" should "leave one frame and no ambient to the renderer" in:
    val renderer = mock[OptiXRenderer]
    (renderer.setRenderConfig _).expects(*).once()
    (renderer.setDenoisingEnabled _).expects(false).once()
    (renderer.setAccumulationFrames _).expects(*).never()
    (renderer.setAmbient _).expects(*).never()

    ReinitRestore.FrameSettings(RenderConfig(), DenoiseMode.Off, 1, None).applyTo(renderer)
