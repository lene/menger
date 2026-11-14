package menger.engines

import menger.common.CausticsConfig
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.config.ExecutionConfig
import menger.config.TAnimationConfig
import menger.dsl.Ambient
import menger.dsl.DenoiseMode
import menger.dsl.Scene
import menger.engines.scene.SceneBuilder
import menger.optix.CameraState
import menger.optix.SceneConfigurator
import menger.spec.ObjectSpec

class AnimationEngine(
  val sceneFunction: Float => Scene,
  val animConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  override val renderConfig: RenderConfig,
  val causticsConfig: CausticsConfig,
  denoiseModeOverride: Option[DenoiseMode] = None,
  ambientOverride: Option[Ambient] = None,
  userSetMaxInstances: Boolean = false
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithAnimation with SavesScreenshots:

  // Sized like the window: with the fixed budget an edge scene the window shows was refused
  // ("requires N instances but limit is 64"), so it could not be exported (msa#56).
  override protected def computeEffectiveMaxInstances(
    builder: SceneBuilder,
    specs: List[ObjectSpec]
  ): Int = autoAdjustedMaxInstances(builder, specs, userSetMaxInstances)

  override protected def textureDir: String = executionConfig.textureDir

  // Evaluate first frame eagerly to initialise the environment (lights, camera, planes)
  private val _firstScene = sceneFunction(animConfig.tForFrame(0))

  override protected val firstFrameConfigs: SceneConverter.SceneConfigs =
    SceneConverter.convert(_firstScene, causticsConfig)

  override protected def denoiseMode: DenoiseMode =
    denoiseModeOverride.getOrElse(firstFrameConfigs.denoiseMode)

  override protected def accumulationFrames: Int = firstFrameConfigs.accumulationFrames

  // The command line's --ambient wins over the scene's, for every frame.
  override protected def ambientFor(configs: SceneConverter.SceneConfigs): Option[Ambient] =
    ambientOverride.orElse(configs.ambient)

  override protected def ambient: Option[Ambient] = ambientFor(firstFrameConfigs)

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

  override protected def currentSaveName: Option[String] =
    Some(String.format(animConfig.savePattern, Integer.valueOf(frameCounter.get())))

  override protected def allowUniformRender: Boolean = executionConfig.allowUniformRender
