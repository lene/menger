package menger.engines

import menger.ProfilingConfig
import menger.config.ExecutionConfig
import menger.dsl.Scene
import menger.dsl.SceneConverter
import menger.input.GdxRuntime
import menger.input.LibGDXInputAdapter
import menger.input.PreviewKeyHandler
import menger.optix.CameraState
import menger.optix.CausticsConfig
import menger.optix.RenderConfig
import menger.optix.SceneConfigurator

class PreviewEngine(
  val sceneFunction: Float => Scene,
  val previewConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  val renderConfig: RenderConfig,
  val causticsConfig: CausticsConfig
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithPreview:

  override protected def textureDir: String = executionConfig.textureDir

  private val _firstScene = sceneFunction(previewConfig.startT)

  override protected val firstFrameConfigs: SceneConverter.SceneConfigs =
    SceneConverter.convert(_firstScene, causticsConfig)

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    firstFrameConfigs.camera.position,
    firstFrameConfigs.camera.lookAt,
    firstFrameConfigs.camera.up,
    firstFrameConfigs.lights
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

  override def render(): Unit = super.render()
