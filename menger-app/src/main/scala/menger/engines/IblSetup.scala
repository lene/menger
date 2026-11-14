package menger.engines

import io.github.lene.optix.OptiXRenderer
import menger.config.IblSunExclusion

/** Image-based lighting state both render paths (the animation frames, the single frame and the
  * interactive window) push to the renderer, so that the two cannot drift apart: a path that
  * leaves it out renders the env map's sun twice, silently. */
object IblSetup:

  /** Leaves the env map's sun out of the lighting, or counts every texel again when there is no
    * exclusion: the renderer keeps the cone until told otherwise, and a scene or a frame may
    * have lost its sun. */
  def applySunExclusion(renderer: OptiXRenderer, exclusion: Option[IblSunExclusion]): Unit =
    exclusion.fold(renderer.clearIBLSunExclusion())(cone =>
      renderer.setIBLSunExclusion(cone.directionToSun, cone.halfAngleDegrees))
