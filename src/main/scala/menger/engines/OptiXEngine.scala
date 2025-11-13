package menger.engines

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.GDXResources
import menger.OptiXResources
import menger.PlaneSpec
import menger.ProfilingConfig
import menger.RenderState
import menger.RotationProjectionParameters
import menger.optix.OptiXRenderer

@SuppressWarnings(Array("org.wartremover.warts.Throw", "org.wartremover.warts.Var"))
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
  val timeout: Float = 0f,
  saveName: Option[String] = None
)(using profilingConfig: ProfilingConfig) extends MengerEngine(
  spongeType, spongeLevel, rotationProjectionParameters, lines, color,
  faceColor, lineColor, fpsLogIntervalMs
) with TimeoutSupport with LazyLogging with SavesScreenshots:

  private val geometryGenerator: Try[OptiXRenderer => Unit] = spongeType match {
    case "sphere" => Try(_.setSphere(center.x, center.y, center.z, sphereRadius))
    case _ => Failure(UnsupportedOperationException(spongeType))
  }
  private lazy val optiXResources: OptiXResources =
    OptiXResources(geometryGenerator, cameraPos, cameraLookat, cameraUp, planeSpec)
  private lazy val batch: SpriteBatch = new SpriteBatch()
  private var renderState: RenderState = RenderState(None, None, 0, 0)
  private var lastCameraWidth: Int = 0
  private var lastCameraHeight: Int = 0

  protected def drawables: List[ModelInstance] =
    throw new UnsupportedOperationException("OptiXEngine doesn't use drawables")

  protected def gdxResources: GDXResources =
    throw new UnsupportedOperationException("OptiXEngine doesn't use gdxResources")

  override def create(): Unit =
    logger.info(s"Creating OptiXEngine with sphere radius=$sphereRadius, color=$color, ior=$ior, scale=$scale")
    optiXResources.setSphereColor(color.r, color.g, color.b, color.a)
    optiXResources.setIOR(ior)
    optiXResources.setScale(scale)
    optiXResources.initialize()
    if timeout > 0 then startExitTimer(timeout)

  override def render(): Unit =
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)

    val width = Gdx.graphics.getWidth
    val height = Gdx.graphics.getHeight

    // Ensure camera is configured for current window dimensions
    if (width != lastCameraWidth || height != lastCameraHeight) then
      logger.info(s"[OptiXEngine] render: dimensions changed from ${lastCameraWidth}x${lastCameraHeight} to ${width}x${height}, updating camera")
      optiXResources.updateCameraAspectRatio(width, height)
      lastCameraWidth = width
      lastCameraHeight = height

    val rgbaBytes = optiXResources.renderScene(width, height)
    renderToScreen(rgbaBytes, width, height)
    saveImage()

  protected def currentSaveName: Option[String] = saveName

  private def renderToScreen(rgbaBytes: Array[Byte], width: Int, height: Int): Unit =
    val needsRecreate = width != renderState.width || height != renderState.height

    if needsRecreate then
      renderState.dispose()
      renderState = RenderState(None, None, width, height)

    val (stateWithPixmap, pm) = renderState.ensurePixmap()
    renderState = stateWithPixmap

    renderState.updatePixmap(rgbaBytes, pm)

    val (stateWithTexture, tex) = renderState.ensureTexture(pm)
    renderState = stateWithTexture

    batch.begin()
    batch.draw(tex, 0, 0, width.toFloat, height.toFloat)
    batch.end()

  override def resize(width: Int, height: Int): Unit =
    logger.info(s"[OptiXEngine] resize event: ${width}x${height}")
    optiXResources.updateCameraAspectRatio(width, height)
    logger.info("[OptiXEngine] resize complete")

  override def dispose(): Unit =
    logger.debug("Disposing OptiXEngine")
    renderState.dispose()
    batch.dispose()
    optiXResources.dispose()
