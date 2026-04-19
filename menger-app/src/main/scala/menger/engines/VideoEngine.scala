package menger.engines

import menger.ProfilingConfig
import menger.config.ExecutionConfig
import menger.dsl.Scene
import menger.dsl.SceneConverter
import menger.optix.CameraState
import menger.optix.CausticsConfig
import menger.optix.RenderConfig
import menger.optix.SceneConfigurator

class VideoEngine(
  val sceneFunction: Float => Scene,
  val animConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  val renderConfig: RenderConfig,
  val causticsConfig: CausticsConfig,
  val videoOutputPath: String,
  val videoQuality: Int,
  val keepFrames: Boolean
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithAnimation with WithVideoExport with SavesScreenshots:

  VideoEncoder.checkAvailable(videoOutputPath)

  override protected def textureDir: String = executionConfig.textureDir

  private val _firstScene = sceneFunction(animConfig.tForFrame(0))

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

  override protected def currentSaveName: Option[String] =
    Some(String.format(animConfig.savePattern, Integer.valueOf(frameCounter.get())))
