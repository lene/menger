package menger.config

import menger.common.CausticsConfig
import menger.common.RenderConfig
import menger.dsl.DenoiseMode

/**
 * Complete configuration for OptiXEngine.
 *
 * Aggregates all configuration aspects into a single object to simplify
 * OptiXEngine constructor and make configuration management clearer.
 *
 * @param scene what geometry to render
 * @param camera where to view from
 * @param environment lighting and ground plane
 * @param execution runtime behavior (timeout, logging, etc.)
 * @param render rendering quality settings (shadows, antialiasing, etc.)
 * @param caustics caustics rendering settings (experimental)
 */
case class OptiXEngineConfig(
  scene: SceneConfig,
  camera: CameraConfig,
  environment: EnvironmentConfig,
  execution: ExecutionConfig,
  render: RenderConfig = RenderConfig.Default,
  caustics: CausticsConfig = CausticsConfig.Disabled,
  cross: CrossConfig = CrossConfig.Disabled,
  denoiseMode: DenoiseMode = DenoiseMode.Off,
  accumulationFrames: Int = 1
):
  require(accumulationFrames >= 1, s"accumulationFrames must be >= 1, got $accumulationFrames")

object OptiXEngineConfig:
  /**
   * Default configuration: single sphere at origin, basic settings
   */
  val Default: OptiXEngineConfig = OptiXEngineConfig(
    scene = SceneConfig.Default,
    camera = CameraConfig.Default,
    environment = EnvironmentConfig.Default,
    execution = ExecutionConfig.Default,
    render = RenderConfig.Default,
    caustics = CausticsConfig.Disabled,
    cross = CrossConfig.Disabled,
    denoiseMode = DenoiseMode.Off,
    accumulationFrames = 1
  )

  /**
   * Configuration for testing: minimal setup with short timeout
   */
  val Testing: OptiXEngineConfig = OptiXEngineConfig(
    scene = SceneConfig.Default,
    camera = CameraConfig.Default,
    environment = EnvironmentConfig.Default,
    execution = ExecutionConfig.Testing,
    render = RenderConfig.Default,
    caustics = CausticsConfig.Disabled,
    cross = CrossConfig.Disabled,
    denoiseMode = DenoiseMode.Off,
    accumulationFrames = 1
  )
