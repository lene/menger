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

  /** Every-frame looping playback (`--playback every-frame`, usability review 2026-10, session 4,
    * F95): every frame is rendered, scene time advances one `1 / EveryFrameRate` step per frame
    * but never faster than real time, and the loop restarts at the end. Starts playing. */
  protected def everyFrame: Boolean = false

  /** Seconds without input or reload after which a playing window pauses itself, so a window
    * left open does not hold the GPU (menger#75); 0 disables. */
  protected def idlePauseSeconds: Double = 600.0

  private def playingClockMode: Boolean = realtime || everyFrame

  private val lastActivityNanos = new java.util.concurrent.atomic.AtomicLong(System.nanoTime())
  private val idlePaused = new AtomicBoolean(false)
  private val lastShownT = new AtomicReference[Float](0f)
  private val skippedTotal = new AtomicReference[Float](0f)

  /** Input or a reload happened: the window is not idle. */
  def noteActivity(): Unit =
    lastActivityNanos.set(System.nanoTime())

  /** The current time and whether the animation is playing, for the window state file. */
  protected def playback: (Float, Boolean) = (currentT.get(), isPlaying.get())

  /** The 4D rotation the user added in the window; applied to every frame's 4D objects. */
  protected def interactive4D: Interactive4DOffset = Interactive4DOffset.Zero

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

  // The camera to render a frame with; the scene's own camera by default.
  protected def applyFrameCamera(
    renderer: io.github.lene.optix.OptiXRenderer,
    sceneCamera: menger.config.CameraConfig
  ): Unit =
    cameraState.updateCamera(
      renderer,
      sceneCamera.position,
      sceneCamera.lookAt,
      sceneCamera.up
    )

  // Draws the current t again with whatever scene the engine now holds (a live reload).
  protected def requestRedraw(): Unit =
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
    if nowPlaying && playingClockMode then
      val playedNanos = ((currentT.get() - previewConfig.startT) * WithPreview.NanosPerSecond).toLong
      playStartNanos.set(System.nanoTime() - playedNanos)
    isPlaying.set(nowPlaying)
    if nowPlaying then
      idlePaused.set(false)
      lastActivityNanos.set(System.nanoTime())
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
    val skipped = skippedTotal.get()
    val behind =
      if skipped > 0f then f" | behind real time: ${skipped}%.1f s skipped" else ""
    val idle = if idlePaused.get() then " | paused: idle (Space resumes)" else ""
    GdxRuntime.setWindowTitle(
      f"$windowTitle | t=$t%.3f | frame $frame/${previewConfig.frames}" + failures + behind + idle
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
    if playingClockMode then togglePlay()

  abstract override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)
    val width  = GdxRuntime.width
    val height = GdxRuntime.height

    // Left alone for too long: pause like Space does, which stops the continuous rendering.
    if isPlaying.get() &&
        WithPreview.isIdle(System.nanoTime(), lastActivityNanos.get(), idlePauseSeconds)
    then
      logger.info(s"No input for ${idlePauseSeconds.toInt} s; pausing the animation")
      idlePaused.set(true)
      togglePlay()
      updateTitle()

    if isPlaying.get() && realtime then
      val elapsedSeconds = (System.nanoTime() - playStartNanos.get()) / WithPreview.NanosPerSecond
      val newT = WithPreview.loopedT(elapsedSeconds, previewConfig.startT, previewConfig.endT)
      val skipped = WithPreview.skippedSeconds(lastShownT.getAndSet(newT), newT)
      // The total restarts with each loop; a wrap is `newT` before the last shown time.
      skippedTotal.updateAndGet(total => if newT < currentT.get() then 0f else total + skipped)
      currentT.set(newT)
      updateTitle()
      needsRender.set(true)
    else if isPlaying.get() && everyFrame then
      val elapsedSeconds = (System.nanoTime() - playStartNanos.get()) / WithPreview.NanosPerSecond
      WithPreview.nextEveryFrame(
        currentT.get(), tStep, elapsedSeconds, previewConfig.startT, previewConfig.endT
      ).foreach { (next, wrapped) =>
        currentT.set(next)
        if wrapped then playStartNanos.set(System.nanoTime())
        updateTitle()
        needsRender.set(true)
      }
    else if isPlaying.get() then
      val next = currentT.get() + tStep
      if next >= previewConfig.endT then
        currentT.set(previewConfig.endT)
        togglePlay()
      else
        currentT.set(next)
      updateTitle()
      needsRender.set(true)

    // A camera moved with the mouse asks the resources for a redraw, not this engine.
    val rebuildFrame = needsRender.getAndSet(false) || renderResources.needsRender
    if rebuildFrame && width > 0 && height > 0 then
      val t = currentT.get()
      Try(sceneFunction(t)) match
        case Failure(e) =>
          reportFailedFrame(t, e)
        case scala.util.Success(dslScene) =>
          val converted = SceneConverter.convert(dslScene, causticsConfig)
          // The rotation added in the window (Shift+drag) rides on every frame (session 4, F96).
          val configs =
            if interactive4D.isZero then converted
            else converted.copy(scene = converted.scene.copy(
              objectSpecs = converted.scene.objectSpecs.map(interactive4D.applyTo)
            ))
          val renderer = rendererWrapper.renderer
          configs.render.foreach(renderer.setRenderConfig)
          renderer.setDenoisingEnabled(configs.denoiseMode == menger.dsl.DenoiseMode.Final)
          if configs.accumulationFrames > 1 then
            renderer.setAccumulationFrames(configs.accumulationFrames)
          val instancesCleared = updateOrRebuild(configs, renderer, t)
          // A builder may have reinitialized the renderer, discarding lights and render
          // settings (usability review 2026-09, F22) -- restore them every frame, with this
          // frame's lights: an animated scene's lights may move (F65).
          sceneConfigurator.configureLights(renderer, configs.lights.toArray)
          renderer.setRenderConfig(configs.render.getOrElse(renderConfig))
          // Planes are real IAS instances: re-add them only when the instances were cleared,
          // or one more is added every in-place frame until the table is full (F58).
          if instancesCleared then PlaneConfigurer.configurePlanes(renderer, configs.planes.toArray)
          configs.background.foreach(c => sceneConfigurator.setBackgroundColor(renderer, c))
          configs.fog.foreach(f => sceneConfigurator.setFog(renderer, f))
          applyFrameCamera(renderer, configs.camera)
          cameraState.updateCameraAspectRatio(renderer, ImageSize(width, height))
          rendererWrapper.renderScene(ImageSize(width, height)) match
            case Some(rgbaBytes) => renderResources.renderToScreen(rgbaBytes, width, height)
            case None            => () // render failed (logged); skip this frame
    // Paused or between key presses: the buffer was cleared above and nothing new was built, so
    // draw the last frame again instead of showing a black window (usability session 3, F68a).
    else if width > 0 && height > 0 then renderResources.redrawExisting(width, height)

  /** A frame that differs from the previous one only in 4D projection or fractional level is
    * applied in place (TrackedMesh4D); anything else is rebuilt, tracked when the scene is
    * 4D-only triangle meshes so the next frame can be updated in place again. A failure is
    * reported as a failed frame. Returns whether all instances were cleared. */
  private def updateOrRebuild(
    configs: SceneConverter.SceneConfigs,
    renderer: io.github.lene.optix.OptiXRenderer,
    t: Float
  ): Boolean =
    val specs = configs.scene.objectSpecs.getOrElse(List.empty)
    val (updated, instancesCleared) =
      tracked4D.get.filter(state => scene.TrackedMesh4D.canUpdateInPlace(state.specs, specs)) match
        case Some(state) =>
          val inPlace = Try(
            tracked4D.set(Some(scene.TrackedMesh4D.updateInPlace(state, specs, renderer)))
          )
          (inPlace, false)
        case None =>
          renderer.clearAllInstances()
          tracked4D.set(None)
          val rebuilt =
            if WithAnimation.is4DOnlyTriangleMeshScene(specs) then
              scene.TrackedMesh4D
                .build(specs, renderer, textureDir, computeEffectiveMaxInstances(_, specs))(using
                  profilingConfig)
                .map(tracked4D.set)
            else buildSceneFromConfigs(configs, renderer)
          (rebuilt, true)
    updated.recover { case e: Exception => reportFailedFrame(t, e) }
    instancesCleared

object WithPreview:
  val NanosPerSecond: Double = 1e9

  /** t for real-time looping playback: `elapsedSeconds` after the start, wrapped into
    * [startT, endT). A non-positive span pins t to startT. */
  def loopedT(elapsedSeconds: Double, startT: Float, endT: Float): Float =
    val span = (endT - startT).toDouble
    if span <= 0.0 then startT else startT + (elapsedSeconds % span).toFloat

  /** Frame rate of the every-frame playback (one `1 / EveryFrameRate` s step per frame). */
  val EveryFrameRate = 25

  /** A time step larger than this between two rendered frames of the real-time playback means
    * frames were dropped (the window skipped scene time, usability review 2026-10, session 4,
    * F95: a 2 s frame made the rest of the loop vanish). */
  val SkipThresholdSeconds = 0.25f

  /** Seconds of scene time skipped between two consecutive rendered frames, 0 for a normal
    * step and at a loop wrap (`newT` before `prevT`). */
  def skippedSeconds(prevT: Float, newT: Float): Float =
    if newT > prevT && newT - prevT > SkipThresholdSeconds then newT - prevT else 0f

  /** The next frame time of the every-frame playback, or `None` while it is not due: every frame
    * is rendered (a slow frame slows the playback instead of skipping scene time), but never
    * faster than real time -- `elapsedSeconds` is the wall time since `t` was in step with it.
    * Returns the new time and whether it wrapped to `startT` at the end of the loop. */
  def nextEveryFrame(
    t: Float, step: Float, elapsedSeconds: Double, startT: Float, endT: Float
  ): Option[(Float, Boolean)] =
    if step <= 0f || (t - startT + step) > elapsedSeconds then None
    else
      val next = t + step
      if next >= endT then Some((startT, true)) else Some((next, false))

  /** Whether a playing window has been left alone long enough to pause itself. */
  def isIdle(nowNanos: Long, lastActivityNanos: Long, idleSeconds: Double): Boolean =
    idleSeconds > 0.0 && (nowNanos - lastActivityNanos) / NanosPerSecond > idleSeconds
