package menger.engines

import java.util.concurrent.atomic.AtomicInteger

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.AnimationSpecificationSequence
import menger.ProfilingConfig
import menger.Vector3Extensions.toVector3
import menger.common.ImageSize
import menger.config.OptiXEngineConfig
import menger.input.GdxRuntime
import menger.optix.CameraState
import menger.optix.RenderConfig
import menger.optix.SceneConfigurator

class CliAnimationEngine(
  config: OptiXEngineConfig,
  animSpec: AnimationSpecificationSequence,
  savePattern: String
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

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    camera.position.toVector3, camera.lookAt.toVector3, camera.up.toVector3,
    environment.lights.toArray
  )

  override protected val cameraState: CameraState =
    CameraState(camera.position.toVector3, camera.lookAt.toVector3, camera.up.toVector3)

  override protected def currentSaveName: Option[String] =
    Some(String.format(savePattern, Integer.valueOf(frameCounter.get())))

  override def create(): Unit =
    logger.info(s"CliAnimationEngine: $totalFrames frames, ${baseSpecs.length} objects")
    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    PlaneConfigurer.configurePlanes(renderer, environment.planes.toArray)
    sceneConfigurator.configureCamera(renderer)
    val firstSpecs = baseSpecs.map(spec => animSpec.applyToSpec(spec, 0))
    buildSceneFromSpecs(firstSpecs, renderer).recover { case e =>
      logger.error(s"Failed to create initial frame: ${e.getMessage}", e)
      GdxRuntime.exit()
    }.get
    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(config.caustics)
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
      renderer.clearAllInstances()
      buildSceneFromSpecs(animatedSpecs, renderer).recover { case e =>
        logger.error(s"Failed to build frame $frame: ${e.getMessage}", e)
      }
      cameraState.updateCameraAspectRatio(renderer, ImageSize(width, height))
      val rgbaBytes = rendererWrapper.renderScene(ImageSize(width, height))
      renderResources.renderToScreen(rgbaBytes, width, height)
      saveImage()
      frameCounter.incrementAndGet()
      ()
    else if frame >= totalFrames then
      logger.info(s"Animation complete: $totalFrames frames rendered")
      GdxRuntime.exit()
