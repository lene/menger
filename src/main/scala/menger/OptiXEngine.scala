package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.typesafe.scalalogging.LazyLogging
import menger.optix.OptiXRenderer

/**
 * Rendering engine that uses OptiX ray tracing to render spheres.
 *
 * This engine works fundamentally differently from Interactive/AnimatedMengerEngine:
 * - OptiX renders directly to 2D RGBA image (like taking a photograph)
 * - Image is displayed in LibGDX window via texture
 * - LibGDX is used only for window management and event handling, not 3D rendering
 * - No geometry/mesh abstractions - OptiX handles ray tracing internally
 *
 * @param sphereRadius Radius of the sphere to render
 * @param timeout Optional timeout in seconds for automated testing (0 = no timeout)
 * @param other parameters Same as MengerEngine base class
 */
// TODO: Replace var with immutable state management (e.g., State monad, Ref, or AtomicReference)
// Current mutable state is for LibGDX/OptiX lifecycle management
// TODO: Replace throw with Either/Try for better error handling
// Current throw is to signal misuse of abstract method that should never be called
@SuppressWarnings(Array("org.wartremover.warts.Var", "org.wartremover.warts.Throw"))
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
) with LazyLogging:

  private var renderer: Option[OptiXRenderer] = None
  private val optiXResources = new OptiXResources()
  private var startTime: Long = 0L

  // TODO: improve engine design - OptiXEngine doesn't use 3D models
  // OptiX renders directly to 2D image, bypassing the Geometry abstraction
  protected def drawables: List[ModelInstance] = Nil

  // OptiX manages its own resources, not standard GDXResources
  protected def gdxResources: GDXResources =
    throw new UnsupportedOperationException(
      "OptiXEngine doesn't use GDXResources - it manages rendering via OptiXResources"
    )

  override def create(): Unit =
    logger.info(s"Creating OptiXEngine with sphere radius=$sphereRadius, color=$color")

    // Check library loaded (forces companion object initialization)
    if !OptiXRenderer.isLibraryLoaded then
      logger.error("OptiX native library failed to load")
      logger.error("Make sure CUDA and OptiX are properly installed")
      logger.error("Build with ENABLE_OPTIX_JNI=true to enable OptiX support")
      System.exit(1)

    // Initialize OptiX renderer
    val r = new OptiXRenderer()
    if !r.isAvailable then
      logger.error("OptiX is not available on this system")
      logger.error("Make sure CUDA and OptiX are properly installed")
      logger.error("Build with ENABLE_OPTIX_JNI=true to enable OptiX support")
      System.exit(1)

    if !r.initialize() then
      logger.error("Failed to initialize OptiX renderer")
      System.exit(1)

    renderer = Some(r)

    // Configure sphere (position at origin)
    r.setSphere(0f, 0f, 0f, sphereRadius)
    logger.debug(s"Configured sphere: center=(0,0,0), radius=$sphereRadius")

    // Configure fixed camera
    // Eye at (0, 0, 3) looking at origin, Y-axis up, 45° FOV
    val eye = Array(0f, 0f, 3f)
    val lookAt = Array(0f, 0f, 0f)
    val up = Array(0f, 1f, 0f)
    val fov = 45f
    r.setCamera(eye, lookAt, up, fov)
    logger.debug(s"Configured camera: eye=${eye.mkString(",")}, lookAt=${lookAt.mkString(",")}, fov=$fov")

    // Configure default lighting
    // Direction from top-right-front, intensity 1.0
    val lightDirection = Array(-1f, -1f, -1f)
    val lightIntensity = 1.0f
    r.setLight(lightDirection, lightIntensity)
    logger.debug(s"Configured light: direction=${lightDirection.mkString(",")}, intensity=$lightIntensity")

    // Initialize rendering resources
    optiXResources.initialize()

    // Start timeout timer if specified
    if timeout > 0 then
      startTime = System.currentTimeMillis()
      logger.info(s"Timeout set to $timeout seconds")

  override def render(): Unit =
    // Check timeout
    val shouldRender = if timeout > 0 then
      val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
      if elapsed >= timeout then
        logger.info(s"Timeout reached ($timeout seconds), exiting")
        Gdx.app.exit()
        false
      else
        true
    else
      true

    if shouldRender then
      // Clear screen (background color #404040 = RGB 64,64,64)
      Gdx.gl.glClearColor(0.25f, 0.25f, 0.25f, 1.0f)
      Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)

      // Render with OptiX
      renderer.foreach { r =>
        val width = Gdx.graphics.getWidth
        val height = Gdx.graphics.getHeight

        // Get rendered image from OptiX
        val rgbaBytes = r.render(width, height)

        // Display image via texture
        optiXResources.updateAndRender(rgbaBytes, width, height)
      }

  override def resize(width: Int, height: Int): Unit =
    logger.debug(s"Window resized to ${width}x${height}")
    optiXResources.resize(width, height)

  override def dispose(): Unit =
    logger.info("Disposing OptiXEngine")
    renderer.foreach(_.dispose())
    renderer = None
    optiXResources.dispose()
