package menger.engines

import java.util.concurrent.atomic.AtomicInteger

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.common.ImageSize
import menger.dsl.Scene
import menger.dsl.SceneConverter
import menger.input.GdxRuntime
import menger.optix.CausticsConfig
import menger.optix.RenderConfig

trait WithAnimation extends RenderEngine with SavesScreenshots with LazyLogging:
  self: BaseEngine =>

  protected def sceneFunction: Float => Scene
  protected def animConfig: TAnimationConfig
  protected def renderConfig: RenderConfig
  protected def causticsConfig: CausticsConfig
  protected def firstFrameConfigs: SceneConverter.SceneConfigs

  protected val frameCounter: AtomicInteger = new AtomicInteger(0)

  abstract override def create(): Unit =
    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configureCamera(renderer)
    buildSceneFromConfigs(firstFrameConfigs, renderer).recover { case e: Exception =>
      logger.error(s"Failed to create initial scene: ${e.getMessage}", e)
      GdxRuntime.exit()
    }.get
    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(firstFrameConfigs.caustics)
    sceneConfigurator.configurePlanes(renderer, firstFrameConfigs.planes)
    GdxRuntime.setContinuousRendering(true)

  abstract override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)
    val width  = GdxRuntime.width
    val height = GdxRuntime.height
    val frame  = frameCounter.get()

    if width > 0 && height > 0 && frame < animConfig.frames then
      val t = animConfig.tForFrame(frame)
      logger.info(s"Rendering frame ${frame + 1}/${animConfig.frames} (t=$t)")
      Try(sceneFunction(t)) match
        case Failure(e) =>
          logger.error(
            s"Scene function threw for frame ${frame + 1}/${animConfig.frames}" +
            s" (t=$t): ${e.getMessage}",
            e
          )
          frameCounter.incrementAndGet()
        case scala.util.Success(dslScene) =>
          val configs  = SceneConverter.convert(dslScene, causticsConfig)
          val renderer = rendererWrapper.renderer
          renderer.clearAllInstances()
          buildSceneFromConfigs(configs, renderer).recover { case e: Exception =>
            logger.error(s"Failed to build scene for frame $frame (t=$t): ${e.getMessage}", e)
          }
          sceneConfigurator.configurePlanes(renderer, configs.planes)
          configs.background.foreach(c => sceneConfigurator.setBackgroundColor(renderer, c))
          cameraState.updateCamera(
            renderer, configs.camera.position, configs.camera.lookAt, configs.camera.up
          )
          cameraState.updateCameraAspectRatio(renderer, ImageSize(width, height))
          val rgbaBytes = rendererWrapper.renderScene(ImageSize(width, height))
          renderResources.renderToScreen(rgbaBytes, width, height)
          saveImage()
          frameCounter.incrementAndGet()
          ()
    else if frame >= animConfig.frames then
      logger.info(s"Animation complete: ${animConfig.frames} frames rendered")
      GdxRuntime.exit()
