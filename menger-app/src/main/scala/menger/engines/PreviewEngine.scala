package menger.engines

import java.util.concurrent.atomic.AtomicReference

import io.github.lene.optix.CameraState
import io.github.lene.optix.SceneConfigurator
import menger.ObjectSpec
import menger.common.CausticsConfig
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.config.CameraConfig
import menger.config.ExecutionConfig
import menger.config.TAnimationConfig
import menger.dsl.DenoiseMode
import menger.dsl.LoadedScene
import menger.dsl.Scene
import menger.dsl.SceneFileWatcher
import menger.dsl.SceneLoader
import menger.engines.scene.SceneBuilder
import menger.input.EventDispatcher
import menger.input.GdxRuntime
import menger.input.LibGDXInputAdapter
import menger.input.OptiXCameraHandler
import menger.input.PreviewKeyHandler
import menger.input.Vector3Extensions.toGdxVector3
import menger.input.Vector3Extensions.toVector3

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
  watchScenePath: Option[java.io.File] = None
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithPreview
    with TimeoutSupport:

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
  private val fileWatcher = new AtomicReference[Option[SceneFileWatcher]](None)

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
      EventDispatcher()
    )

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
    GdxRuntime.setInputProcessor(LibGDXInputAdapter(Seq(keyHandler, cameraController)))
    startExitTimer(timeout)
    watchScenePath.foreach(startWatchingSceneFile)

  override def render(): Unit = super.render()

  override def dispose(): Unit =
    fileWatcher.get().foreach(_.close())
    super.dispose()

  // Runs on the watcher's own thread; the swap itself happens on the GL thread. A bad edit is
  // logged and the playing scene keeps running -- it must never kill the window.
  private def startWatchingSceneFile(file: java.io.File): Unit =
    fileWatcher.set(Some(new SceneFileWatcher(file)(() => onSceneFileChanged(file))))
    logger.info(s"Watching scene file for live reload: ${file.getPath}")

  private def onSceneFileChanged(file: java.io.File): Unit =
    PreviewEngine.reloadDecision(SceneLoader.load(file.getPath), previewConfig, realtime) match
      case PreviewEngine.Reload(fn, config) =>
        GdxRuntime.postRunnable(() => reloadScene(fn, config))
      case PreviewEngine.KindChanged =>
        logger.warn(
          s"${file.getPath} changed to a static scene; the animated window cannot show it -- " +
          "restart the window to pick it up"
        )
      case PreviewEngine.Rejected(error) =>
        logger.warn(s"Failed to reload ${file.getPath}, keeping the current scene: $error")

  // GL thread. The preview takes its camera from the scene on every frame, so a camera edit
  // shows without further handling (F64).
  private def reloadScene(fn: Float => Scene, config: TAnimationConfig): Unit =
    scala.util.Try(stateFor(fn, config)) match
      case scala.util.Success(newState) =>
        state.set(newState)
        logger.info(s"Reloaded animated scene from file (duration ${config.endT}s)")
        requestRedraw()
      case scala.util.Failure(e) =>
        logger.warn(s"Failed to reload the animated scene, keeping the current one: ${e.getMessage}")

object PreviewEngine:

  // What a changed scene file means for the playing window.
  sealed trait ReloadDecision
  case class Reload(sceneFunction: Float => Scene, previewConfig: TAnimationConfig)
      extends ReloadDecision
  case object KindChanged extends ReloadDecision
  case class Rejected(error: String) extends ReloadDecision

  // A real-time scene's new `duration` becomes its new time range; a `--preview` scene keeps
  // the range it was started with.
  def reloadDecision(
    loaded: Either[String, LoadedScene],
    current: TAnimationConfig,
    realtime: Boolean
  ): ReloadDecision =
    loaded match
      case Right(animated @ LoadedScene.Animated(fn)) =>
        val config =
          if realtime then current.copy(endT = animated.duration.getOrElse(current.endT))
          else current
        Reload(fn, config)
      case Right(_: LoadedScene.Static) => KindChanged
      case Left(error)                  => Rejected(error)
