package menger.engines

import java.util.concurrent.atomic.AtomicInteger

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.CameraState
import io.github.lene.optix.SceneConfigurator
import menger.AnimationSpecificationSequence
import menger.common.ImageSize
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.config.OptiXEngineConfig
import menger.dsl.DenoiseMode
import menger.input.GdxRuntime

object CliAnimationEngine:
  def formatSaveName(savePattern: Option[String], frame: Int): Option[String] =
    savePattern.map(p => String.format(p, Integer.valueOf(frame)))

class CliAnimationEngine(
  config: OptiXEngineConfig,
  animSpec: AnimationSpecificationSequence,
  savePattern: Option[String]
)(using ProfilingConfig)
    extends BaseEngine(config.execution.maxInstances)
    with SavesScreenshots with LazyLogging:

  private val execution   = config.execution
  private val camera      = config.camera
  private val environment = config.environment

  private val baseSpecs   = config.scene.objectSpecs.getOrElse(List.empty)
  private val totalFrames = animSpec.numFrames
  private val frameCounter = new AtomicInteger(0)

  override protected def textureDir: String       = execution.textureDir
  override protected def renderConfig: RenderConfig = config.render
  override protected def denoiseMode: DenoiseMode = config.denoiseMode
  override protected def accumulationFrames: Int  = config.accumulationFrames

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    camera.position, camera.lookAt, camera.up,
    environment.lights.toArray
  )

  override protected val cameraState: CameraState =
    CameraState(camera.position, camera.lookAt, camera.up)

  override protected def currentSaveName: Option[String] =
    CliAnimationEngine.formatSaveName(savePattern, frameCounter.get())

  override def create(): Unit =
    logger.info(s"CliAnimationEngine: $totalFrames frames, ${baseSpecs.length} objects")
    val renderer = rendererWrapper.renderer
    val firstSpecs = baseSpecs.map(spec => animSpec.applyToSpec(spec, 0))
    // Build before configuring lights/planes/camera, not after: a scene builder (e.g.
    // TesseractEdgeSceneBuilder, for edge-heavy 4D objects) may call renderer.reinitialize()
    // when its own instance-count check exceeds the constructor-time budget, and
    // reinitialize disposes and recreates the native handle — dropping any plane/light state
    // set on it beforehand (Sprint 36 H2.1, same fix as InteractiveEngine.create()).
    buildSceneFromSpecs(firstSpecs, renderer).recover { case e =>
      logger.error(s"Failed to create initial frame: ${e.getMessage}", e)
      GdxRuntime.exit()
    }.get
    sceneConfigurator.configureLights(renderer)
    PlaneConfigurer.configurePlanes(renderer, environment.planes.toArray)
    sceneConfigurator.configureCamera(renderer)
    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(config.caustics)
    configureOutputMode(renderer)
    environment.background.foreach(c => sceneConfigurator.setBackgroundColor(renderer, c))
    environment.fog.foreach(f => sceneConfigurator.setFog(renderer, f))
    GdxRuntime.setContinuousRendering(true)

  override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)
    val width  = GdxRuntime.width
    val height = GdxRuntime.height
    val frame  = frameCounter.get()

    if width > 0 && height > 0 && frame < totalFrames then
      logger.info(s"Rendering frame ${frame + 1}/$totalFrames")
      val animatedSpecs = baseSpecs.map(spec => animSpec.applyToSpec(spec, frame))
      val renderer = rendererWrapper.renderer
      configureOutputMode(renderer)
      renderer.clearAllInstances()
      buildSceneFromSpecs(animatedSpecs, renderer).recover { case e =>
        logger.error(s"Failed to build frame $frame: ${e.getMessage}", e)
      }
      // Planes are real IAS instances (Sprint 36 H3.1) — clearAllInstances above wiped
      // them too, so they must be re-added every frame, not just at create().
      PlaneConfigurer.configurePlanes(renderer, environment.planes.toArray)
      cameraState.updateCameraAspectRatio(renderer, ImageSize(width, height))
      rendererWrapper.renderScene(ImageSize(width, height)) match
        case Some(rgbaBytes) => renderResources.renderToScreen(rgbaBytes, width, height)
        case None            => () // render failed (logged); skip this frame
      saveImage()
      frameCounter.incrementAndGet()
      ()
    else if frame >= totalFrames then
      logger.info(s"Animation complete: $totalFrames frames rendered")
      GdxRuntime.exit()
