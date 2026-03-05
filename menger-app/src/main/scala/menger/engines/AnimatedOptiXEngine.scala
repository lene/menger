package menger.engines

import java.util.concurrent.atomic.AtomicInteger

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.OptiXRenderResources
import menger.ProfilingConfig
import menger.common.ImageSize
import menger.config.ExecutionConfig
import menger.dsl.Scene
import menger.dsl.SceneConverter
import menger.engines.scene.SphereSceneBuilder
import menger.gdx.GdxRuntime
import menger.optix.CameraState
import menger.optix.CausticsConfig
import menger.optix.OptiXRendererWrapper
import menger.optix.RenderConfig
import menger.optix.SceneConfigurator

/** Engine that renders a t-parameter animation: evaluates scene(t) per frame,
  * rebuilds the OptiX scene, renders, and saves. Exits after the last frame.
  *
  * @param sceneFunction The animated scene function (t => Scene)
  * @param animConfig Animation parameters (startT, endT, frames, savePattern)
  * @param executionConfig Runtime settings (maxInstances, etc.)
  * @param renderConfig Rendering quality settings
  * @param causticsConfig Caustics settings
  */
class AnimatedOptiXEngine(
  sceneFunction: Float => Scene,
  animConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  renderConfig: RenderConfig,
  causticsConfig: CausticsConfig
)(using profilingConfig: ProfilingConfig)
  extends RenderEngine with LazyLogging with SavesScreenshots:

  private val frameCounter = AtomicInteger(0)
  private val rendererWrapper = OptiXRendererWrapper(executionConfig.maxInstances)
  private val renderResources: OptiXRenderResources = OptiXRenderResources(0, 0)

  // Evaluate first frame to initialize environment
  private val firstScene = sceneFunction(animConfig.tForFrame(0))
  private val firstConfigs = SceneConverter.convert(firstScene, causticsConfig)

  private val sceneConfigurator = SceneConfigurator(
    Failure(UnsupportedOperationException("Legacy geometry generator not used in animated engine")),
    firstConfigs.camera.position,
    firstConfigs.camera.lookAt,
    firstConfigs.camera.up,
    firstConfigs.lights
  )

  private val cameraState = CameraState(
    firstConfigs.camera.position, firstConfigs.camera.lookAt, firstConfigs.camera.up
  )

  override def create(): Unit =
    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configureCamera(renderer)

    // Build the first frame's scene
    val configs = firstConfigs
    buildSceneFromConfigs(configs).recover { case e: Exception =>
      logger.error(s"Failed to create initial scene: ${e.getMessage}", e)
      GdxRuntime.exit()
    }.get

    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(configs.caustics)
    sceneConfigurator.configurePlanes(renderer, configs.planes)

    GdxRuntime.setContinuousRendering(true)

  override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)

    val width = GdxRuntime.width
    val height = GdxRuntime.height
    val frame = frameCounter.get()

    if width > 0 && height > 0 && frame < animConfig.frames then
      val t = animConfig.tForFrame(frame)
      logger.info(s"Rendering frame ${frame + 1}/${animConfig.frames} (t=$t)")

      // Evaluate scene(t) — wrap in Try so a throwing scene function skips the frame
      Try(sceneFunction(t)) match
        case Failure(e) =>
          logger.error(
            s"Scene function threw for frame ${frame + 1}/${animConfig.frames} (t=$t): ${e.getMessage}", e
          )
          frameCounter.incrementAndGet()
        case scala.util.Success(dslScene) =>
          val configs = SceneConverter.convert(dslScene, causticsConfig)

          // Rebuild geometry
          val renderer = rendererWrapper.renderer
          renderer.clearAllInstances()
          buildSceneFromConfigs(configs).recover { case e: Exception =>
            logger.error(s"Failed to build scene for frame $frame (t=$t): ${e.getMessage}", e)
          }

          // Reconfigure planes per frame (supports animated plane changes)
          sceneConfigurator.configurePlanes(renderer, configs.planes)

          // Apply per-scene background color if set
          configs.background.foreach(c => sceneConfigurator.setBackgroundColor(renderer, c))

          // Update camera from this frame's scene
          cameraState.updateCamera(
            renderer, configs.camera.position, configs.camera.lookAt, configs.camera.up
          )
          cameraState.updateCameraAspectRatio(renderer, ImageSize(width, height))

          // Render
          val rgbaBytes = rendererWrapper.renderScene(ImageSize(width, height))
          renderResources.renderToScreen(rgbaBytes, width, height)

          // Save frame
          saveImage()

          frameCounter.incrementAndGet()
          ()
    else if frame >= animConfig.frames then
      logger.info(s"Animation complete: ${animConfig.frames} frames rendered")
      GdxRuntime.exit()

  private def buildSceneFromConfigs(configs: SceneConverter.SceneConfigs): Try[Unit] =
    val specs = configs.scene.objectSpecs.getOrElse(List.empty)
    val sceneType = SceneClassifier.classify(specs)

    sceneType match
      case SceneType.Spheres(_) =>
        SphereSceneBuilder().buildScene(specs, rendererWrapper.renderer, executionConfig.maxInstances)
      case SceneType.TriangleMeshes(_) =>
        SceneClassifier.selectSceneBuilder(sceneType, Some(executionConfig.textureDir))
          .get.buildScene(specs, rendererWrapper.renderer, executionConfig.maxInstances)
      case SceneType.SimpleMixed(allSpecs, _) =>
        Try {
          val sphereSpecs = allSpecs.filter(_.objectType.toLowerCase == "sphere")
          val meshSpecs   = allSpecs.filterNot(_.objectType.toLowerCase == "sphere")
          if sphereSpecs.nonEmpty then
            SphereSceneBuilder()
              .buildScene(sphereSpecs, rendererWrapper.renderer, executionConfig.maxInstances).get
          if meshSpecs.nonEmpty then
            SceneClassifier.selectSceneBuilder(
              SceneType.TriangleMeshes(meshSpecs), Some(executionConfig.textureDir)
            ).get.buildScene(meshSpecs, rendererWrapper.renderer, executionConfig.maxInstances).get
        }
      case other =>
        SceneClassifier.selectSceneBuilder(other, Some(executionConfig.textureDir)) match
          case Some(builder) =>
            builder.buildScene(specs, rendererWrapper.renderer, executionConfig.maxInstances)
          case None =>
            Failure(UnsupportedOperationException(s"Unsupported scene type: $other"))

  // SavesScreenshots implementation -- format pattern with current frame index
  override protected def currentSaveName: Option[String] =
    Some(String.format(animConfig.savePattern, Integer.valueOf(frameCounter.get())))

  override def resize(width: Int, height: Int): Unit = {}
  override def dispose(): Unit =
    logger.debug("Disposing AnimatedOptiXEngine")
    renderResources.dispose()
    rendererWrapper.dispose()
  override def pause(): Unit = {}
  override def resume(): Unit = {}
