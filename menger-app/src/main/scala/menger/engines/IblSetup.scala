package menger.engines

import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.OptiXRenderer
import menger.config.IblSunExclusion
import menger.engines.scene.TextureManager
import menger.video.EnvMapVideo

/** Image-based lighting state every render path (the animation frames, the preview, the CLI
  * animation, the single frame and the interactive window) pushes to the renderer, so that they
  * cannot drift apart: a path that leaves it out renders without the sky (menger#139) or the env
  * map's sun twice, silently. */
object IblSetup extends LazyLogging:

  /** Uploads the static environment map and makes it the background, when there is one. */
  def uploadEnvMap(renderer: OptiXRenderer, textureDir: String, envMap: Option[String]): Unit =
    envMap.foreach { path =>
      TextureManager.uploadFile(renderer, textureDir, path) match
        case Right(idx)   => renderer.setEnvironmentMap(idx)
        case Left(reason) => logger.error(s"Failed to load environment map: $path: $reason")
    }

  /** Uploads the environment map video's frame at `renderT` and makes it the background. */
  def uploadEnvMapVideo(
    renderer: OptiXRenderer,
    textureDir: String,
    envMapVideo: Option[EnvMapVideo],
    renderT: Float
  ): Unit =
    envMapVideo.foreach(video =>
      TextureManager.loadInitialEnvMapVideo(video, renderer, textureDir, renderT)
        .foreach(renderer.setEnvironmentMap))

  /** The environment map, then the env-map video, as the window sets them after every build: a
    * builder's `renderer.reinitialize` and a rebuild both leave the renderer without them. */
  def uploadEnvMaps(
    renderer: OptiXRenderer,
    textureDir: String,
    envMap: Option[String],
    envMapVideo: Option[EnvMapVideo],
    renderT: Float
  ): Unit =
    uploadEnvMap(renderer, textureDir, envMap)
    uploadEnvMapVideo(renderer, textureDir, envMapVideo, renderT)

  /** What the renderer was last given as its sky, and the capacity it had then: a builder's
    * `reinitialize` grows the capacity and drops the textures, so a changed capacity means the
    * sky is gone. */
  final case class Sky(
    envMap: Option[String], envMapVideo: Option[EnvMapVideo], rendererCapacity: Int
  )

  /** Sets the sky again only when the renderer does not hold it: it was never set, the scene's
    * env map or video changed, or the renderer was reinitialized since. A plain rebuild keeps the
    * uploaded texture, and re-reading an HDR (and recomputing its CDF) on every step of an
    * interactive 4D rotation is a visible stall. Returns what the renderer holds now. */
  def applySky(
    renderer: OptiXRenderer,
    textureDir: String,
    envMap: Option[String],
    envMapVideo: Option[EnvMapVideo],
    renderT: Float,
    applied: Option[Sky]
  ): Option[Sky] =
    val wanted = Sky(envMap, envMapVideo, renderer.maxInstances)
    if !applied.contains(wanted) then
      uploadEnvMaps(renderer, textureDir, envMap, envMapVideo, renderT)
    Some(wanted)

  /** The scene's image-based lighting; an env-map video also sets it (off, unless asked for). */
  def applySceneIBL(renderer: OptiXRenderer, configs: SceneConverter.SceneConfigs): Unit =
    if configs.iblEnabled || configs.envMapVideo.nonEmpty then
      renderer.setIBL(
        enabled  = configs.iblEnabled,
        strength = configs.iblStrength,
        samples  = configs.iblSamples
      )
      applySunExclusion(renderer, configs.iblSunExclusion)

  /** The scene's env map and image-based lighting, as the animation export sets them. */
  def applyEnvironment(
    renderer: OptiXRenderer,
    textureDir: String,
    configs: SceneConverter.SceneConfigs
  ): Unit =
    uploadEnvMap(renderer, textureDir, configs.envMap)
    applySceneIBL(renderer, configs)

  /** Leaves the env map's sun out of the lighting, or counts every texel again when there is no
    * exclusion: the renderer keeps the cone until told otherwise, and a scene or a frame may
    * have lost its sun. */
  def applySunExclusion(renderer: OptiXRenderer, exclusion: Option[IblSunExclusion]): Unit =
    exclusion.fold(renderer.clearIBLSunExclusion())(cone =>
      renderer.setIBLSunExclusion(cone.directionToSun, cone.halfAngleDegrees))
