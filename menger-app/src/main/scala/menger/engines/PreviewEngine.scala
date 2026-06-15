package menger.engines

import io.github.lene.optix.CameraState
import io.github.lene.optix.SceneConfigurator
import menger.Vector3Extensions.toVector3
import menger.common.CausticsConfig
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.config.ExecutionConfig
import menger.config.TAnimationConfig
import menger.dsl.DenoiseMode
import menger.dsl.Scene
import menger.input.GdxRuntime
import menger.input.LibGDXInputAdapter
import menger.input.PreviewKeyHandler

class PreviewEngine(
  val sceneFunction: Float => Scene,
  val previewConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  override val renderConfig: RenderConfig,
  val causticsConfig: CausticsConfig,
  denoiseModeOverride: Option[DenoiseMode] = None
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithPreview:

  override protected def textureDir: String = executionConfig.textureDir

  private val _firstScene = sceneFunction(previewConfig.startT)

  override protected val firstFrameConfigs: SceneConverter.SceneConfigs =
    SceneConverter.convert(_firstScene, causticsConfig)

  override protected def denoiseMode: DenoiseMode =
    denoiseModeOverride.getOrElse(firstFrameConfigs.denoiseMode)

  override protected def accumulationFrames: Int = firstFrameConfigs.accumulationFrames

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    firstFrameConfigs.camera.position.toVector3,
    firstFrameConfigs.camera.lookAt.toVector3,
    firstFrameConfigs.camera.up.toVector3,
    firstFrameConfigs.lights.toArray
  )

  override protected val cameraState: CameraState = CameraState(
    firstFrameConfigs.camera.position.toVector3,
    firstFrameConfigs.camera.lookAt.toVector3,
    firstFrameConfigs.camera.up.toVector3
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

  override def render(): Unit = super.render()
