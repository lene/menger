package menger.engines

import menger.Vector3Extensions.toVector3
import menger.common.CausticsConfig
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.config.ExecutionConfig
import menger.config.TAnimationConfig
import menger.dsl.Scene
import menger.optix.CameraState
import menger.optix.SceneConfigurator

class VideoEngine(
  val sceneFunction: Float => Scene,
  val animConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  override val renderConfig: RenderConfig,
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

  override protected def currentSaveName: Option[String] =
    Some(String.format(animConfig.savePattern, Integer.valueOf(frameCounter.get())))

  override protected def allowUniformRender: Boolean = executionConfig.allowUniformRender
