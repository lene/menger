package menger.engines

import java.util.concurrent.atomic.AtomicInteger

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.OptiXRenderResources
import menger.ProfilingConfig
import menger.cli.PlaneColorSpec
import menger.cli.PlaneSpec
import menger.common.ImageSize
import menger.config.ExecutionConfig
import menger.dsl.Scene
import menger.dsl.SceneConverter
import menger.engines.scene.SceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder
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
  * @param planeSpec Ground plane configuration
  * @param planeColor Optional plane color
  */
class AnimatedOptiXEngine(
  sceneFunction: Float => Scene,
  animConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  renderConfig: RenderConfig,
  causticsConfig: CausticsConfig,
  planeSpec: PlaneSpec,
  planeColor: Option[PlaneColorSpec]
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
    planeSpec,
    firstConfigs.lights
  )

  private val cameraState = CameraState(
    firstConfigs.camera.position, firstConfigs.camera.lookAt, firstConfigs.camera.up
  )

  override def create(): Unit =
    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configurePlane(renderer)
    sceneConfigurator.configureCamera(renderer)

    // Build the first frame's scene
    val configs = firstConfigs
    buildSceneFromConfigs(configs).recover { case e: Exception =>
      logger.error(s"Failed to create initial scene: ${e.getMessage}", e)
      GdxRuntime.exit()
    }.get

    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(configs.caustics)
    planeColor.foreach(sceneConfigurator.setPlaneColor(renderer, _))

    GdxRuntime.setContinuousRendering(true)

  override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)

    val width = GdxRuntime.width
    val height = GdxRuntime.height
    val frame = frameCounter.get()

    if width > 0 && height > 0 && frame < animConfig.frames then
      val t = animConfig.tForFrame(frame)
      logger.info(s"Rendering frame ${frame + 1}/${animConfig.frames} (t=$t)")

      // Evaluate scene(t) and rebuild
      val dslScene = sceneFunction(t)
      val configs = SceneConverter.convert(dslScene, causticsConfig)

      // Rebuild geometry
      val renderer = rendererWrapper.renderer
      renderer.clearAllInstances()
      buildSceneFromConfigs(configs).recover { case e: Exception =>
        logger.error(s"Failed to build scene for frame $frame (t=$t): ${e.getMessage}", e)
      }

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
    val sceneType = classifyScene(specs)

    sceneType match
      case SceneType.Spheres(_) =>
        SphereSceneBuilder().buildScene(specs, rendererWrapper.renderer, executionConfig.maxInstances)
      case SceneType.TriangleMeshes(_) =>
        TriangleMeshSceneBuilder(executionConfig.textureDir)(using profilingConfig)
          .buildScene(specs, rendererWrapper.renderer, executionConfig.maxInstances)
      case SceneType.SimpleMixed(allSpecs, _) =>
        Try {
          val sphereSpecs = allSpecs.filter(_.objectType.toLowerCase == "sphere")
          val meshSpecs = allSpecs.filterNot(_.objectType.toLowerCase == "sphere")
          if sphereSpecs.nonEmpty then
            SphereSceneBuilder().buildScene(sphereSpecs, rendererWrapper.renderer, executionConfig.maxInstances).get
          if meshSpecs.nonEmpty then
            TriangleMeshSceneBuilder(executionConfig.textureDir)(using profilingConfig)
              .buildScene(meshSpecs, rendererWrapper.renderer, executionConfig.maxInstances).get
        }
      case other =>
        selectSceneBuilder(other) match
          case Some(builder) =>
            builder.buildScene(specs, rendererWrapper.renderer, executionConfig.maxInstances)
          case None =>
            Failure(UnsupportedOperationException(s"Unsupported scene type: $other"))

  private def selectSceneBuilder(sceneType: SceneType): Option[SceneBuilder] =
    sceneType match
      case SceneType.Spheres(_) => Some(SphereSceneBuilder())
      case SceneType.TriangleMeshes(_) =>
        Some(TriangleMeshSceneBuilder(executionConfig.textureDir)(using profilingConfig))
      case SceneType.CubeSponges(_) =>
        Some(menger.engines.scene.CubeSpongeSceneBuilder())
      case _ => None

  private def classifyScene(specs: List[menger.ObjectSpec]): SceneType =
    val types = specs.map(_.objectType.toLowerCase).toSet
    if types.contains("cube-sponge") then SceneType.CubeSponges(specs)
    else if types.forall(_ == "sphere") then SceneType.Spheres(specs)
    else if types.forall(isTriangleMeshType) then SceneType.TriangleMeshes(specs)
    else
      val hasSpheres = types.contains("sphere")
      val meshTypes = types.filter(isTriangleMeshType)
      if hasSpheres && meshTypes.nonEmpty then SceneType.SimpleMixed(specs, meshTypes.head)
      else SceneType.ComplexMixed(specs)

  private def isTriangleMeshType(objectType: String): Boolean =
    objectType == "cube" || menger.common.ObjectType.isSponge(objectType) ||
      menger.common.ObjectType.isProjected4D(objectType)

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
