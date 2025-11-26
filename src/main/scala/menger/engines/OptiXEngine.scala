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
import menger.optix.CameraState
import menger.optix.OptiXRenderer
import menger.optix.OptiXRendererWrapper
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
  val shadows: Boolean = false,
  val lights: Option[List[menger.LightSpec]] = None,
  val antialiasing: Boolean = false,
  val aaMaxDepth: Int = 2,
  val aaThreshold: Float = 0.1f,
  val caustics: Boolean = false,
  val causticsPhotons: Int = 100000,
  val causticsIterations: Int = 10,
  val causticsRadius: Float = 0.1f,
  val causticsAlpha: Float = 0.7f
)(using profilingConfig: ProfilingConfig) extends RenderEngine with TimeoutSupport with LazyLogging with SavesScreenshots:

  private val geometryGenerator: Try[OptiXRenderer => Unit] = spongeType match {
    case "sphere" => Try(_.setSphere(menger.common.Vector[3](center.x, center.y, center.z), sphereRadius))
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
    logger.info(s"Creating OptiXEngine with sphere radius=$sphereRadius, color=$color, ior=$ior, scale=$scale, shadows=$shadows, antialiasing=$antialiasing, caustics=$caustics")

    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureScene(renderer)
    sceneConfigurator.setSphereColor(renderer, color.toCommonColor)
    sceneConfigurator.setIOR(renderer, ior)
    sceneConfigurator.setScale(renderer, scale)
    sceneConfigurator.setShadows(renderer, shadows)
    sceneConfigurator.setAntialiasing(renderer, antialiasing, aaMaxDepth, aaThreshold)
    sceneConfigurator.setCaustics(renderer, caustics, causticsPhotons, causticsIterations, causticsRadius, causticsAlpha)
    planeColor.foreach(sceneConfigurator.setPlaneColor(renderer, _))

    // Register interactive camera controller for mouse-based camera control
    Gdx.input.setInputProcessor(cameraController)
    logger.info("Interactive camera controls enabled (left-click: orbit, right-click: pan, scroll: zoom)")

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
      // Check if window dimensions changed
      val dimensionsChanged = renderResources.currentDimensions match
        case Some(lastDims) => width != lastDims.width || height != lastDims.height
        case None => true  // First valid render, dimensions definitely changed

      if dimensionsChanged then
        renderResources.currentDimensions match
          case Some(lastDims) =>
            logger.info(s"[OptiXEngine] render: dimensions changed from ${lastDims.width}x${lastDims.height} to ${width}x${height}, updating camera")
          case None =>
            logger.info(s"[OptiXEngine] render: initializing with dimensions ${width}x${height}")
        cameraState.updateCameraAspectRatio(rendererWrapper.renderer, ImageSize(width, height))
        renderResources.markNeedsRender()

      // Only render the scene if something changed
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

  override def resize(width: Int, height: Int): Unit =
    logger.info(s"[OptiXEngine] resize event: ${width}x${height}")
    cameraState.updateCameraAspectRatio(rendererWrapper.renderer, ImageSize(width, height))
    renderResources.markNeedsRender()
    Gdx.graphics.requestRendering()
    logger.info("[OptiXEngine] resize complete")

  override def dispose(): Unit =
    logger.debug("Disposing OptiXEngine")
    renderResources.dispose()
    rendererWrapper.dispose()

  override def pause(): Unit = {}
  override def resume(): Unit = {}
