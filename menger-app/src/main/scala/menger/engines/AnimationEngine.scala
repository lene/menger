package menger.engines

import scala.util.Failure

import menger.ProfilingConfig
import menger.config.ExecutionConfig
import menger.dsl.Scene
import menger.dsl.SceneConverter
import menger.optix.CameraState
import menger.optix.CausticsConfig
import menger.optix.RenderConfig
import menger.optix.SceneConfigurator

class AnimationEngine(
  val sceneFunction: Float => Scene,
  val animConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  val renderConfig: RenderConfig,
  val causticsConfig: CausticsConfig
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithAnimation with SavesScreenshots:

  override protected def textureDir: String = executionConfig.textureDir

  // Evaluate first frame eagerly to initialise the environment (lights, camera, planes)
  private val _firstScene = sceneFunction(animConfig.tForFrame(0))

  override protected val firstFrameConfigs: SceneConverter.SceneConfigs =
    SceneConverter.convert(_firstScene, causticsConfig)

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    Failure(UnsupportedOperationException("Legacy geometry generator not used in animated engine")),
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
