package menger.engines

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.ColorConversions.toCommonColor
import menger.OptiXRenderResources
import menger.PlaneColorSpec
import menger.PlaneSpec
import menger.ProfilingConfig
import menger.common.ImageSize
import menger.input.OptiXCameraController
import menger.input.OptiXInputMultiplexer
import menger.objects.Cube
import menger.objects.SpongeBySurface
import menger.objects.SpongeByVolume
import menger.optix.CameraState
import menger.optix.CausticsConfig
import menger.optix.OptiXRenderer
import menger.optix.OptiXRendererWrapper
import menger.optix.RenderConfig
import menger.optix.SceneConfigurator

class OptiXEngine(
  val spongeType: String,
  val spongeLevel: Float,
  val lines: Boolean,
  val color: Color,
  val fpsLogIntervalMs: Int,
  val sphereRadius: Float,
  val ior: Float,
  val scale: Float,
  val cameraPos: Vector3,
  val cameraLookat: Vector3,
  val cameraUp: Vector3,
  val center: Vector3,
  val planeSpec: PlaneSpec,
  val planeColor: Option[PlaneColorSpec] = None,
  val timeout: Float = 0f,
  saveName: Option[String] = None,
  val enableStats: Boolean = false,
  val lights: Option[List[menger.LightSpec]] = None,
  val renderConfig: RenderConfig = RenderConfig.Default,
  val causticsConfig: CausticsConfig = CausticsConfig.Disabled
)(using profilingConfig: ProfilingConfig) extends RenderEngine with TimeoutSupport with LazyLogging with SavesScreenshots:

  // Level thresholds for warnings (based on triangle counts and performance)
  private val VolumeLevelWarning = 3  // Level 3 = ~32K cubes = ~384K triangles
  private val SurfaceLevelWarning = 3 // Level 3 = ~10K triangles per face = ~60K total
  private val VolumeLevelMax = 5      // Level 5 = ~3.2M cubes (may exhaust GPU memory)
  private val SurfaceLevelMax = 5     // Level 5 = ~31M triangles (may exhaust GPU memory)

  private def warnIfHighLevel(): Unit =
    val intLevel = spongeLevel.toInt
    spongeType match
      case "sponge-volume" =>
        if intLevel >= VolumeLevelWarning then
          val estimatedTriangles = math.pow(20, intLevel).toLong * 12 // 20^level cubes * 12 triangles each
          logger.warn(s"Sponge level $intLevel may be slow (~${estimatedTriangles / 1000}K triangles)")
        if intLevel > VolumeLevelMax then
          logger.error(s"Sponge level $intLevel exceeds recommended maximum ($VolumeLevelMax)")
      case "sponge-surface" =>
        if intLevel >= SurfaceLevelWarning then
          val estimatedTriangles = math.pow(12, intLevel).toLong * 6 * 2 // 12^level sub-faces * 6 faces * 2 triangles
          logger.warn(s"Sponge level $intLevel may be slow (~${estimatedTriangles / 1000}K triangles)")
        if intLevel > SurfaceLevelMax then
          logger.error(s"Sponge level $intLevel exceeds recommended maximum ($SurfaceLevelMax)")
      case _ => // No warning for other types

  private val geometryGenerator: Try[OptiXRenderer => Unit] = spongeType match {
    case "sphere" => Try(_.setSphere(menger.common.Vector[3](center.x, center.y, center.z), sphereRadius))
    case "cube" => Try { renderer =>
      val cube = Cube(center = center, scale = sphereRadius * 2)  // Use radius as half-size for consistency
      val mesh = cube.toTriangleMesh
      renderer.setTriangleMesh(mesh)
    }
    case "sponge-volume" => Try { renderer =>
      val sponge = SpongeByVolume(center = center, scale = sphereRadius * 2, level = spongeLevel)
      val mesh = sponge.toTriangleMesh
      renderer.setTriangleMesh(mesh)
    }
    case "sponge-surface" => Try { renderer =>
      given menger.ProfilingConfig = profilingConfig
      val sponge = SpongeBySurface(center = center, scale = sphereRadius * 2, level = spongeLevel)
      val mesh = sponge.toTriangleMesh
      renderer.setTriangleMesh(mesh)
    }
    case _ => Failure(UnsupportedOperationException(spongeType))
  }

  // Composition: Three focused components instead of one god object
  private val rendererWrapper = OptiXRendererWrapper()
  private val sceneConfigurator = SceneConfigurator(geometryGenerator, cameraPos, cameraLookat, cameraUp, planeSpec, lights)
  private val cameraState = CameraState(cameraPos, cameraLookat, cameraUp)

  private val renderResources: OptiXRenderResources = OptiXRenderResources(0, 0)
  private lazy val cameraController: OptiXCameraController =
    OptiXCameraController(rendererWrapper, cameraState, renderResources, cameraPos, cameraLookat, cameraUp)

  override def create(): Unit =
    logger.info(s"Creating OptiXEngine with object=$spongeType, radius=$sphereRadius, color=$color, ior=$ior, scale=$scale, renderConfig=$renderConfig, causticsConfig=$causticsConfig")

    // Warn about high sponge levels before generating geometry
    warnIfHighLevel()

    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureScene(renderer)

    // Configure color and IOR based on object type
    spongeType match
      case "sphere" =>
        sceneConfigurator.setSphereColor(renderer, color.toCommonColor)
        sceneConfigurator.setIOR(renderer, ior)
      case "cube" | "sponge-volume" | "sponge-surface" =>
        sceneConfigurator.setTriangleMeshColor(renderer, color.toCommonColor)
        sceneConfigurator.setTriangleMeshIOR(renderer, ior)
      case _ =>
        // For other types, try both (backward compatibility)
        sceneConfigurator.setSphereColor(renderer, color.toCommonColor)
        sceneConfigurator.setIOR(renderer, ior)

    sceneConfigurator.setScale(renderer, scale)
    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(causticsConfig)
    planeColor.foreach(sceneConfigurator.setPlaneColor(renderer, _))

    // Register input multiplexer for mouse-based camera control and keyboard shortcuts
    Gdx.input.setInputProcessor(OptiXInputMultiplexer(cameraController))

    // Disable continuous rendering - we'll request renders only when needed
    Gdx.graphics.setContinuousRendering(false)
    Gdx.graphics.requestRendering()

    if timeout > 0 then startExitTimer(timeout)

  override def render(): Unit =
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)

    val width = Gdx.graphics.getWidth
    val height = Gdx.graphics.getHeight

    // Only proceed if window dimensions are valid
    if width > 0 && height > 0 then
      // Initialize camera on first render (window is not resizable in OptiX mode)
      if renderResources.currentDimensions.isEmpty then
        cameraState.updateCameraAspectRatio(rendererWrapper.renderer, ImageSize(width, height))
        renderResources.markNeedsRender()

      // Only render the scene if something changed (camera moved, etc.)
      if renderResources.needsRender then
        val size = ImageSize(width, height)
        val rgbaBytes = if enableStats then renderWithStats(width, height) else rendererWrapper.renderScene(size)
        renderResources.renderToScreen(rgbaBytes, width, height)
      else
        // Just redraw the existing texture without re-rendering
        renderResources.redrawExisting(width, height)

      saveImage()

    // Exit after saving when in non-interactive mode (unless timeout is set)
    if saveName.isDefined && !renderResources.hasSaved && timeout == 0 then
      renderResources.markSaved()
      Gdx.app.exit()

  protected def currentSaveName: Option[String] = saveName

  private def renderWithStats(width: Int, height: Int): Array[Byte] =
    val result = rendererWrapper.renderSceneWithStats(ImageSize(width, height))
    val stats = result.stats
    logger.info(
      s"Ray stats: primary=${stats.primaryRays} total=${stats.totalRays} " +
      s"reflected=${stats.reflectedRays} refracted=${stats.refractedRays} " +
      s"shadow=${stats.shadowRays} aa=${stats.aaRays} " +
      s"depth=${stats.minDepthReached}-${stats.maxDepthReached}"
    )
    result.image

  // Window resize is disabled for OptiX mode (setResizable(false) in Main)
  // This method is kept for interface compatibility but should never be called with different dimensions
  override def resize(width: Int, height: Int): Unit = {}

  override def dispose(): Unit =
    logger.debug("Disposing OptiXEngine")
    renderResources.dispose()
    rendererWrapper.dispose()

  override def pause(): Unit = {}
  override def resume(): Unit = {}
