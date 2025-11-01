package menger.engines

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.typesafe.scalalogging.LazyLogging
import menger.GDXResources
import menger.OptiXResources
import menger.ProfilingConfig
import menger.RotationProjectionParameters
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
  val timeout: Float = 0f
)(using profilingConfig: ProfilingConfig) extends MengerEngine(
  spongeType, spongeLevel, rotationProjectionParameters, lines, color,
  faceColor, lineColor, fpsLogIntervalMs
) with TimeoutSupport with LazyLogging:

  private lazy val renderer: OptiXRenderer = initializeRenderer
  private lazy val optiXResources: OptiXResources = new OptiXResources(renderer, sphereRadius)

  protected def drawables: List[ModelInstance] =
    throw new UnsupportedOperationException("OptiXEngine doesn't use drawables")

  protected def gdxResources: GDXResources =
    throw new UnsupportedOperationException("OptiXEngine doesn't use gdxResources")

  private def errorExit(message: String): Unit =
    logger.error(message)
    System.exit(1)

  override def create(): Unit =
    logger.info(s"Creating OptiXEngine with sphere radius=$sphereRadius, color=$color")
    optiXResources.initialize()
    if timeout > 0 then startExitTimer(timeout)

  private def initializeRenderer: OptiXRenderer =
    // Check library loaded (forces companion object initialization)
    if !OptiXRenderer.isLibraryLoaded then
      errorExit("OptiX native library failed to load - ensure CUDA and OptiX are available")

    // Initialize OptiX renderer
    val renderer = new OptiXRenderer()
    if !renderer.isAvailable then
      errorExit("OptiX not available on this system - ensure CUDA and OptiX are available")

    if !renderer.initialize() then
      errorExit("Failed to initialize OptiX renderer")

    renderer

  override def render(): Unit =
    Gdx.gl.glClearColor(0.25f, 0.25f, 0.25f, 1.0f)
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)

    val width = Gdx.graphics.getWidth
    val height = Gdx.graphics.getHeight
    val rgbaBytes = renderer.render(width, height)
    optiXResources.render(rgbaBytes, width, height)

  override def resize(width: Int, height: Int): Unit =
    logger.debug(s"Window resized to ${width}x${height}")
    optiXResources.resize(width, height)

  override def dispose(): Unit =
    logger.info("Disposing OptiXEngine")
    optiXResources.dispose()
    renderer.dispose()
