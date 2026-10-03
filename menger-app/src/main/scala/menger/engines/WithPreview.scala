package menger.engines

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.common.CausticsConfig
import menger.common.ImageSize
import menger.common.RenderConfig
import menger.config.TAnimationConfig
import menger.dsl.Scene
import menger.input.GdxRuntime

trait WithPreview extends RenderEngine with LazyLogging:
  self: BaseEngine =>

  protected def sceneFunction: Float => Scene
  protected def previewConfig: TAnimationConfig
  protected def renderConfig: RenderConfig
  protected def causticsConfig: CausticsConfig
  protected def firstFrameConfigs: SceneConverter.SceneConfigs
  protected def windowTitle: String = "Menger Sponges"

  /** Real-time looping playback, for a scene that declares its `duration`: t follows the wall
    * clock through [startT, endT) and wraps around, instead of advancing one tStep per
    * rendered frame. Starts playing immediately. */
  protected def realtime: Boolean = false

  private val currentT    = new AtomicReference[Float](0f)
  private val isPlaying   = new AtomicBoolean(false)
  private val needsRender = new AtomicBoolean(true)
  private val playStartNanos = new AtomicReference[Long](0L)
  // The previous frame's 4D scene with its renderer handles, when it can be updated in place
  // (usability review 2026-09, session 2, F52: every frame used to rebuild the whole scene).
  private val tracked4D = new AtomicReference[Option[scene.TrackedMesh4D.State]](None)
  // Frames whose scene failed to build -- the window then shows the previous frame, so say so
  // in the title (menger#54).
  private val failedFrames = new AtomicInteger(0)

  private def reportFailedFrame(t: Float, e: Throwable): Unit =
    failedFrames.incrementAndGet()
    logger.error(FrameBuildFailure.message(s"t=$t", e), e)
    updateTitle()

  private def tStep: Float =
    val range = previewConfig.endT - previewConfig.startT
    if previewConfig.frames > 1 then range / (previewConfig.frames - 1) else range

  def stepT(delta: Float): Unit =
    val clamped = clampT(currentT.get() + delta)
    currentT.set(clamped)
    updateTitle()
    needsRender.set(true)
    GdxRuntime.requestRendering()

  def jumpToStart(): Unit =
    currentT.set(previewConfig.startT)
    updateTitle()
    needsRender.set(true)
    GdxRuntime.requestRendering()

  def jumpToEnd(): Unit =
    currentT.set(previewConfig.endT)
    updateTitle()
    needsRender.set(true)
    GdxRuntime.requestRendering()

  def togglePlay(): Unit =
    val nowPlaying = !isPlaying.get()
    // Resuming real-time playback continues from the current t, not from the start.
    if nowPlaying && realtime then
      val playedNanos = ((currentT.get() - previewConfig.startT) * WithPreview.NanosPerSecond).toLong
      playStartNanos.set(System.nanoTime() - playedNanos)
    isPlaying.set(nowPlaying)
    GdxRuntime.setContinuousRendering(nowPlaying)
    if nowPlaying then GdxRuntime.requestRendering()

  private def clampT(v: Float): Float =
    math.max(previewConfig.startT, math.min(previewConfig.endT, v))

  private def frameForT(t: Float): Int =
    if tStep > 0 then math.round((t - previewConfig.startT) / tStep) else 0

  private def updateTitle(): Unit =
    val t     = currentT.get()
    val frame = frameForT(t)
    val failed = failedFrames.get()
    val failures = if failed > 0 then s" | $failed frame(s) failed to build" else ""
    GdxRuntime.setWindowTitle(
      f"$windowTitle | t=$t%.3f | frame $frame/${previewConfig.frames}" + failures
    )

  abstract override def create(): Unit =
    currentT.set(previewConfig.startT)
    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configureCamera(renderer)
    buildSceneFromConfigs(firstFrameConfigs, renderer).recover { case e: Exception =>
      logger.error(s"Failed to create initial preview scene: ${e.getMessage}", e)
      GdxRuntime.exit()
    }.get
    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(firstFrameConfigs.caustics)
    configureOutputMode(renderer)
    PlaneConfigurer.configurePlanes(renderer, firstFrameConfigs.planes.toArray)
    GdxRuntime.setContinuousRendering(false)
    updateTitle()
    if realtime then togglePlay()

  abstract override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)
    val width  = GdxRuntime.width
    val height = GdxRuntime.height

    if isPlaying.get() && realtime then
      val elapsedSeconds = (System.nanoTime() - playStartNanos.get()) / WithPreview.NanosPerSecond
      currentT.set(WithPreview.loopedT(elapsedSeconds, previewConfig.startT, previewConfig.endT))
      updateTitle()
      needsRender.set(true)
    else if isPlaying.get() then
      val next = currentT.get() + tStep
      if next >= previewConfig.endT then
        currentT.set(previewConfig.endT)
        togglePlay()
      else
        currentT.set(next)
      updateTitle()
      needsRender.set(true)

    if needsRender.getAndSet(false) && width > 0 && height > 0 then
      val t = currentT.get()
      Try(sceneFunction(t)) match
        case Failure(e) =>
          reportFailedFrame(t, e)
        case scala.util.Success(dslScene) =>
          val configs  = SceneConverter.convert(dslScene, causticsConfig)
          val renderer = rendererWrapper.renderer
          configs.render.foreach(renderer.setRenderConfig)
          renderer.setDenoisingEnabled(configs.denoiseMode == menger.dsl.DenoiseMode.Final)
          if configs.accumulationFrames > 1 then
            renderer.setAccumulationFrames(configs.accumulationFrames)
          updateOrRebuild(configs, renderer).recover { case e: Exception =>
            reportFailedFrame(t, e)
          }
          // A builder may have reinitialized the renderer, discarding lights and render
          // settings (usability review 2026-09, F22) -- restore them every frame.
          sceneConfigurator.configureLights(renderer)
          renderer.setRenderConfig(configs.render.getOrElse(renderConfig))
          PlaneConfigurer.configurePlanes(renderer, configs.planes.toArray)
          configs.background.foreach(c => sceneConfigurator.setBackgroundColor(renderer, c))
          configs.fog.foreach(f => sceneConfigurator.setFog(renderer, f))
          cameraState.updateCamera(
            renderer,
            configs.camera.position,
            configs.camera.lookAt,
            configs.camera.up
          )
          cameraState.updateCameraAspectRatio(renderer, ImageSize(width, height))
          rendererWrapper.renderScene(ImageSize(width, height)) match
            case Some(rgbaBytes) => renderResources.renderToScreen(rgbaBytes, width, height)
            case None            => () // render failed (logged); skip this frame

  /** A frame that differs from the previous one only in 4D projection or fractional level is
    * applied in place (TrackedMesh4D); anything else is rebuilt, tracked when the scene is
    * 4D-only triangle meshes so the next frame can be updated in place again. */
  private def updateOrRebuild(
    configs: SceneConverter.SceneConfigs,
    renderer: io.github.lene.optix.OptiXRenderer
  ): Try[Unit] =
    val specs = configs.scene.objectSpecs.getOrElse(List.empty)
    tracked4D.get.filter(state => scene.TrackedMesh4D.canUpdateInPlace(state.specs, specs)) match
      case Some(state) =>
        Try(tracked4D.set(Some(scene.TrackedMesh4D.updateInPlace(state, specs, renderer))))
      case None =>
        renderer.clearAllInstances()
        tracked4D.set(None)
        if WithAnimation.is4DOnlyTriangleMeshScene(specs) then
          scene.TrackedMesh4D
            .build(specs, renderer, textureDir, computeEffectiveMaxInstances(_, specs))(using
              profilingConfig)
            .map(tracked4D.set)
        else buildSceneFromConfigs(configs, renderer)

object WithPreview:
  val NanosPerSecond: Double = 1e9

  /** t for real-time looping playback: `elapsedSeconds` after the start, wrapped into
    * [startT, endT). A non-positive span pins t to startT. */
  def loopedT(elapsedSeconds: Double, startT: Float, endT: Float): Float =
    val span = (endT - startT).toDouble
    if span <= 0.0 then startT else startT + (elapsedSeconds % span).toFloat
