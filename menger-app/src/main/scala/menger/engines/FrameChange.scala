package menger.engines

import menger.config.PlaneConfig
import menger.spec.ObjectSpec

/** What a new frame changes in the scene the renderer already holds, so that a frame rebuilds
  * only what it must (menger#86). Camera, lights, background and fog are set on every frame
  * anyway; they never need a rebuild. */
enum FrameChange:
  /** Same objects, planes and environment map: nothing to rebuild. */
  case Unchanged
  case Rebuild

object FrameChange:

  /** The part of a frame that its instances are built from. */
  final case class Geometry(
    specs: List[ObjectSpec],
    planes: List[PlaneConfig],
    envMap: Option[String],
    animatedTextures: Boolean
  )

  object Geometry:
    def of(configs: SceneConverter.SceneConfigs): Geometry =
      val specs = configs.scene.objectSpecs.getOrElse(List.empty)
      Geometry(specs, configs.planes, configs.envMap, WithAnimation.hasVideoSources(configs, specs))

  /** A frame with a video texture is never unchanged: its frames advance with t. Equality is
    * conservative: a spec holding a function or a mesh array compares by reference, so a scene
    * function that creates a new one per frame rebuilds, as before. */
  def classify(previous: Option[Geometry], next: Geometry): FrameChange =
    if previous.contains(next) && !next.animatedTextures then Unchanged else Rebuild
