package menger.engines

import io.github.lene.optix.OptiXRenderer
import menger.common.RenderConfig
import menger.dsl.Ambient
import menger.dsl.DenoiseMode

/** What a builder's `renderer.reinitialize` discards (it recreates the native renderer when a
  * frame needs more instances than the renderer holds), set again after the frame's build: the
  * animation went on without its env map and image-based lighting from the first such frame to
  * the end of the process (menger#198; InteractiveEngine does the same in `applyRendererState`
  * and `applyEnvironmentMaps`). The reinitialize is recognised by the changed capacity. */
object ReinitRestore:

  /** The settings a frame sets before its build; the build's reinitialize drops them. */
  final case class FrameSettings(
    render: RenderConfig,
    denoise: DenoiseMode,
    accumulationFrames: Int,
    ambient: Option[Ambient]
  ):
    def applyTo(renderer: OptiXRenderer): Unit =
      renderer.setRenderConfig(render)
      renderer.setDenoisingEnabled(denoise == DenoiseMode.Final)
      if accumulationFrames > 1 then renderer.setAccumulationFrames(accumulationFrames)
      ambient.foreach(a => renderer.setAmbient(AmbientMapping.toModel(a)))

  /** Sets what the renderer lost, or got a new value of, in a frame built with
    * `capacityBefore` instances: after a reinitialize the frame's settings, the caustics and the
    * sky; the static env map (re-read only then, or when it changed) first, then
    * `configureEnvMapVideo`, which replaces it as the background, and the lighting last, since it
    * samples the map the renderer holds. Returns the sky the renderer holds now. */
  def restore(
    renderer: OptiXRenderer,
    textureDir: String,
    configs: SceneConverter.SceneConfigs,
    settings: FrameSettings,
    capacityBefore: Int,
    appliedSky: Option[IblSetup.Sky]
  )(configureEnvMapVideo: => Unit): Option[IblSetup.Sky] =
    val reinitialized = renderer.maxInstances != capacityBefore
    if reinitialized then
      settings.applyTo(renderer)
      renderer.setCausticsConfig(configs.caustics)
    val sky = IblSetup.applySky(
      renderer, textureDir, configs.envMap, None, 0f, appliedSky
    )
    configureEnvMapVideo
    if reinitialized || sky != appliedSky then IblSetup.applySceneIBL(renderer, configs)
    sky
