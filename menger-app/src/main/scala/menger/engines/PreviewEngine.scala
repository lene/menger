package menger.engines

import io.github.lene.optix.CameraState
import io.github.lene.optix.SceneConfigurator
import menger.ObjectSpec
import menger.common.CausticsConfig
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.config.ExecutionConfig
import menger.config.TAnimationConfig
import menger.dsl.DenoiseMode
import menger.dsl.Scene
import menger.engines.scene.SceneBuilder
import menger.input.GdxRuntime
import menger.input.LibGDXInputAdapter
import menger.input.PreviewKeyHandler

class PreviewEngine(
  val sceneFunction: Float => Scene,
  val previewConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  override val renderConfig: RenderConfig,
  val causticsConfig: CausticsConfig,
  denoiseModeOverride: Option[DenoiseMode] = None,
  override val realtime: Boolean = false,
  userSetMaxInstances: Boolean = false
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithPreview
    with TimeoutSupport:

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

  private val _firstScene = sceneFunction(previewConfig.startT)

  override protected val firstFrameConfigs: SceneConverter.SceneConfigs =
    SceneConverter.convert(_firstScene, causticsConfig)

  override protected def denoiseMode: DenoiseMode =
    denoiseModeOverride.getOrElse(firstFrameConfigs.denoiseMode)

  override protected def accumulationFrames: Int = firstFrameConfigs.accumulationFrames

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    firstFrameConfigs.camera.position,
    firstFrameConfigs.camera.lookAt,
    firstFrameConfigs.camera.up,
    firstFrameConfigs.lights.toArray
  )

  override protected val cameraState: CameraState = CameraState(
    firstFrameConfigs.camera.position,
    firstFrameConfigs.camera.lookAt,
    firstFrameConfigs.camera.up
  )

  override def create(): Unit =
    super.create()
    val keyHandler = PreviewKeyHandler(
      onStep       = stepT,
      onTogglePlay = togglePlay,
      onJumpStart  = jumpToStart,
      onJumpEnd    = jumpToEnd
    )
    GdxRuntime.setInputProcessor(LibGDXInputAdapter(Seq(keyHandler)))
    startExitTimer(timeout)

  override def render(): Unit = super.render()
