package menger.engines

import java.util.concurrent.atomic.AtomicReference

import menger.common.CausticsConfig
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.config.CameraConfig
import menger.config.ExecutionConfig
import menger.config.TAnimationConfig
import menger.dsl.DenoiseMode
import menger.dsl.LoadedScene
import menger.dsl.Scene
import menger.engines.scene.SceneBuilder
import menger.input.EventDispatcher
import menger.input.GdxRuntime
import menger.input.LibGDXInputAdapter
import menger.input.OptiXCameraHandler
import menger.input.PreviewKeyHandler
import menger.input.Vector3Extensions.toGdxVector3
import menger.input.Vector3Extensions.toVector3
import menger.objects.higher_d.RotationProjectionParameters
import menger.optix.CameraState
import menger.optix.SceneConfigurator
import menger.scene.LoadError
import menger.spec.ObjectSpec

class PreviewEngine(
  initialSceneFunction: Float => Scene,
  initialPreviewConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  override val renderConfig: RenderConfig,
  val causticsConfig: CausticsConfig,
  denoiseModeOverride: Option[DenoiseMode] = None,
  override val realtime: Boolean = false,
  userSetMaxInstances: Boolean = false,
  // When set (a real `--scene <file.scala>`), the file is watched and an edit that stays an
  // animated scene replaces the playing scene (usability session 3, F57).
  watchScenePath: Option[java.io.File] = None,
  // Every-frame looping playback instead of real time (`--playback every-frame`, session 4,
  // F95), and the idle pause (menger#75).
  override val everyFrame: Boolean = false,
  idlePause: Double = 600.0
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithPreview
    with TimeoutSupport
    with menger.input.Observer
    with LiveReload:

  // Everything a reload swaps: the scene function, its time range, and what is derived from
  // its first frame (lights, camera, output mode).
  private case class PreviewState(
    sceneFunction: Float => Scene,
    previewConfig: TAnimationConfig,
    firstFrameConfigs: SceneConverter.SceneConfigs,
    sceneConfigurator: SceneConfigurator
  )

  private def stateFor(fn: Float => Scene, config: TAnimationConfig): PreviewState =
    val firstFrame = SceneConverter.convert(fn(config.startT), causticsConfig)
    PreviewState(
      fn,
      config,
      firstFrame,
      SceneConfigurator(
        firstFrame.camera.position,
        firstFrame.camera.lookAt,
        firstFrame.camera.up,
        firstFrame.lights.toArray
      )
    )

  private val state = new AtomicReference[PreviewState](
    stateFor(initialSceneFunction, initialPreviewConfig)
  )

  override protected def sceneFunction: Float => Scene = state.get().sceneFunction
  override def previewConfig: TAnimationConfig = state.get().previewConfig
  override protected def firstFrameConfigs: SceneConverter.SceneConfigs =
    state.get().firstFrameConfigs
  override protected def sceneConfigurator: SceneConfigurator = state.get().sceneConfigurator

  override protected def textureDir: String = executionConfig.textureDir

  // The preview used the fixed budget, so an animated scene with edge cylinders failed at the
  // first frame ("requires N instances but limit is 64"); size it like InteractiveEngine does
  // (usability session 3, F73).
  override protected def computeEffectiveMaxInstances(
    builder: SceneBuilder,
    specs: List[ObjectSpec]
  ): Int = autoAdjustedMaxInstances(builder, specs, userSetMaxInstances)

  // --timeout was ignored here, so a looping real-time preview never ended on its own.
  override def timeout: Float = executionConfig.timeout

  override protected def idlePauseSeconds: Double = idlePause

  override protected def eventsFor: Option[java.io.File] = watchScenePath
  override protected def onStartupFailed(e: Throwable): Unit = startupFailed(e)
  override protected def onFrameFailed(context: String, e: Throwable): Unit = frameFailed(context, e)
  override protected def onFrameSlow(detail: String): Unit = frameSlow(detail)

  override protected def denoiseMode: DenoiseMode =
    denoiseModeOverride.getOrElse(firstFrameConfigs.denoiseMode)

  override protected def accumulationFrames: Int = firstFrameConfigs.accumulationFrames

  override protected val cameraState: CameraState = CameraState(
    firstFrameConfigs.camera.position,
    firstFrameConfigs.camera.lookAt,
    firstFrameConfigs.camera.up
  )

  // Mouse orbit/pan/zoom (usability session 3, F68b). A 4D rotation event has no observer here.
  private lazy val cameraController: OptiXCameraHandler =
    OptiXCameraHandler(
      rendererWrapper,
      cameraState,
      renderResources,
      firstFrameConfigs.camera.position.toGdxVector3,
      firstFrameConfigs.camera.lookAt.toGdxVector3,
      firstFrameConfigs.camera.up.toGdxVector3,
      EventDispatcher().withObserver(this)
    )

  // Shift+drag 4D rotation (usability review 2026-10, session 4, F96): kept as an offset that
  // rides on every frame, and across a reload unless the file changed its own projections.
  private val offset4D = new AtomicReference(Interactive4DOffset.Zero)
  override protected def interactive4D: Interactive4DOffset = offset4D.get()

  override def handleEvent(event: RotationProjectionParameters): Unit =
    val _ = offset4D.updateAndGet(_.plus(event))
    requestRedraw()

  private val lastSceneCamera = new AtomicReference[CameraConfig](firstFrameConfigs.camera)

  // The scene's camera takes over only when it changed since the previous frame (an animated or
  // reloaded camera); otherwise the mouse view stays. A builder may have reinitialized the
  // renderer, so the current view is applied every frame.
  override protected def applyFrameCamera(
    renderer: io.github.lene.optix.OptiXRenderer,
    sceneCamera: CameraConfig
  ): Unit =
    if lastSceneCamera.getAndSet(sceneCamera) != sceneCamera then
      cameraController.setCamera(
        sceneCamera.position.toGdxVector3,
        sceneCamera.lookAt.toGdxVector3,
        sceneCamera.up.toGdxVector3
      )
    cameraState.updateCamera(
      renderer,
      cameraController.currentEye.toVector3,
      cameraController.currentLookAt.toVector3,
      cameraController.currentUp.toVector3
    )

  override def create(): Unit =
    super.create()
    val keyHandler = PreviewKeyHandler(
      onStep       = stepT,
      onTogglePlay = togglePlay,
      onJumpStart  = jumpToStart,
      onJumpEnd    = jumpToEnd
    )
    GdxRuntime.setInputProcessor(LibGDXInputAdapter(
      Seq(menger.input.ActivityHandler(() => noteActivity()), keyHandler, cameraController)
    ))
    startExitTimer(timeout)
    watchScenePath.foreach(startWatchingSceneFile)

  private val stateFile = watchScenePath.map(f => WindowStateFile(WindowStateFile.pathFor(f)))

  override def render(): Unit =
    super.render()
    stateFile.foreach { file =>
      val (t, playing) = playback
      file.update(WindowStateFile.of(
        cameraController.currentEye.toVector3, cameraController.currentLookAt.toVector3,
        cameraController.currentUp.toVector3, offset4D.get(), Some(t), Some(playing)
      ))
    }

  override def dispose(): Unit =
    stopWatchingSceneFile()
    super.dispose()

  // Runs on the watcher's own thread; the swap itself happens on the GL thread. A bad edit is
  // logged and the playing scene keeps running -- it must never kill the window.
  private def startWatchingSceneFile(file: java.io.File): Unit =
    watchSceneFile(file) { loaded =>
      PreviewEngine.reloadDecision(loaded, previewConfig, realtime, everyFrame) match
        case PreviewEngine.Reload(fn, config) =>
          GdxRuntime.postRunnable(() => reloadScene(file, fn, config))
        case PreviewEngine.KindChanged     => kindChanged(file, toAnimated = false)
        case PreviewEngine.Rejected(failure) => reloadFailed(file, failure)
    }

  // GL thread. The preview takes its camera from the scene on every frame, so a camera edit
  // shows without further handling (F64). The new scene's first frame is checked before the
  // swap (review 2026-10-14, menger#101): a scene that cannot build used to replace the playing
  // one and then fail on every frame.
  private def reloadScene(file: java.io.File, fn: Float => Scene, config: TAnimationConfig): Unit =
    noteActivity() // an edit is not idleness
    val checked = scala.util.Try(stateFor(fn, config)).flatMap(newState =>
      precheck(newState.firstFrameConfigs.scene.objectSpecs.getOrElse(Nil)).map(_ => newState))
    checked match
      case scala.util.Success(newState) =>
        def fileProjections(s: PreviewState) =
          Interactive4DOffset.fileProjections(s.firstFrameConfigs.scene.objectSpecs.getOrElse(Nil))
        if fileProjections(state.get()) != fileProjections(newState) then
          offset4D.set(Interactive4DOffset.Zero)
        state.set(newState)
        reloaded("animated scene", s"duration ${config.endT}s")
        requestRedraw()
      case scala.util.Failure(e) => reloadFailed(file, SceneFailure.of(e))

object PreviewEngine:

  /** Frames of an every-frame loop of `duration` seconds: one per `1 / EveryFrameRate` s, plus
    * the first. */
  def everyFrameCount(duration: Float): Int =
    math.max(2, math.round(duration * WithPreview.EveryFrameRate) + 1)

  // What a changed scene file means for the playing window.
  sealed trait ReloadDecision
  case class Reload(sceneFunction: Float => Scene, previewConfig: TAnimationConfig)
      extends ReloadDecision
  case object KindChanged extends ReloadDecision
  case class Rejected(failure: SceneFailure) extends ReloadDecision

  // A real-time scene's new `duration` becomes its new time range; a `--preview` scene keeps
  // the range it was started with.
  def reloadDecision(
    loaded: Either[LoadError, LoadedScene],
    current: TAnimationConfig,
    realtime: Boolean,
    everyFrame: Boolean = false
  ): ReloadDecision =
    loaded match
      case Right(animated @ LoadedScene.Animated(fn)) =>
        val config =
          if realtime || everyFrame then
            val endT = animated.duration.getOrElse(current.endT)
            current.copy(endT = endT, frames = if everyFrame then everyFrameCount(endT) else current.frames)
          else current
        Reload(fn, config)
      case Right(_: LoadedScene.Static) => KindChanged
      case Left(error)                  => Rejected(SceneFailure.Load(error))
