package menger.engines

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.ColorConversions.toCommonColor
import menger.GDXResources
import menger.OptiXRenderResources
import menger.OptiXResources
import menger.PlaneColorSpec
import menger.PlaneSpec
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.common.ImageSize
import menger.common.Vector
import menger.input.OptiXCameraController
import menger.optix.OptiXRenderer

@SuppressWarnings(Array("org.wartremover.warts.Throw"))
class OptiXEngine(
  spongeType: String,
  spongeLevel: Float,
  rotationProjectionParameters: RotationProjectionParameters,
  lines: Boolean,
  color: Color,
  faceColor: Option[Color],
  lineColor: Option[Color],
  fpsLogIntervalMs: Int,
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
  val aaThreshold: Float = 0.1f
)(using profilingConfig: ProfilingConfig) extends MengerEngine(
  spongeType, spongeLevel, rotationProjectionParameters, lines, color,
  faceColor, lineColor, fpsLogIntervalMs
) with TimeoutSupport with LazyLogging with SavesScreenshots:

  private val geometryGenerator: Try[OptiXRenderer => Unit] = spongeType match {
    case "sphere" => Try(_.setSphere(Vector[3](center.x, center.y, center.z), sphereRadius))
    case _ => Failure(UnsupportedOperationException(spongeType))
  }
  private lazy val optiXResources: OptiXResources =
    OptiXResources(geometryGenerator, cameraPos, cameraLookat, cameraUp, planeSpec, lights)
  private val renderResources: OptiXRenderResources = OptiXRenderResources(0, 0)
  private lazy val cameraController: OptiXCameraController =
    OptiXCameraController(optiXResources, renderResources, cameraPos, cameraLookat, cameraUp)

  protected def drawables: List[ModelInstance] =
    throw new UnsupportedOperationException("OptiXEngine doesn't use drawables")

  protected def gdxResources: GDXResources =
    throw new UnsupportedOperationException("OptiXEngine doesn't use gdxResources")

  override def create(): Unit =
    logger.info(s"Creating OptiXEngine with sphere radius=$sphereRadius, color=$color, ior=$ior, scale=$scale, shadows=$shadows, antialiasing=$antialiasing")
    optiXResources.setSphereColor(color.toCommonColor)
    optiXResources.setIOR(ior)
    optiXResources.setScale(scale)
    optiXResources.setShadows(shadows)
    optiXResources.setAntialiasing(antialiasing, aaMaxDepth, aaThreshold)
    planeColor.foreach(optiXResources.setPlaneColor)
    optiXResources.initialize()

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
        optiXResources.updateCameraAspectRatio(ImageSize(width, height))
        renderResources.markNeedsRender()

      // Only render the scene if something changed
      if renderResources.needsRender then
        val size = ImageSize(width, height)
        val rgbaBytes = if enableStats then renderWithStats(width, height) else optiXResources.renderScene(size)
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
    val result = optiXResources.renderSceneWithStats(ImageSize(width, height))
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
    optiXResources.updateCameraAspectRatio(ImageSize(width, height))
    renderResources.markNeedsRender()
    Gdx.graphics.requestRendering()
    logger.info("[OptiXEngine] resize complete")

  override def dispose(): Unit =
    logger.debug("Disposing OptiXEngine")
    renderResources.dispose()
    optiXResources.dispose()
