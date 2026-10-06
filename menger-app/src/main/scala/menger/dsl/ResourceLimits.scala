package menger.dsl

import menger.common.Const

/** Central resource ceilings, enforced identically wherever a scene can request them: DSL
  * object constructors' `require`s, the CLI `ObjectSpec` parser, and
  * `InteractiveEngine`'s level table. One number per limit instead of several copies that can
  * drift apart -- a DSL scene had no upper bound at all on sponge/L-system/parametric-surface
  * size, only the CLI path did (usability review 2026-09, T1#3).
  */
object ResourceLimits:

  /** `warnAt`: log a slowness warning at or above this level. `max`: hard reject above this
    * level (`require`/`Left`, not just a log line). */
  case class LevelLimit(warnAt: Int, max: Int)

  val cubeSpongeLevel: LevelLimit =
    LevelLimit(Const.Engine.spongeLevelWarningThreshold, Const.Engine.cubeSpongeMaxLevel)
  val tesseractSpongeVolumeLevel: LevelLimit =
    LevelLimit(Const.Engine.tesseractSpongeWarnLevel, Const.Engine.tesseractSpongeMaxLevel)
  val tesseractSpongeSurfaceLevel: LevelLimit =
    LevelLimit(Const.Engine.tesseractSponge2WarnLevel, Const.Engine.tesseractSponge2MaxLevel)

  /** Keyed by normalized object-type name (`menger.common.ObjectType.normalize`). Shared by
    * `InteractiveEngine`'s level table and `ObjectSpec.validateSpongeLevel`. */
  val levelLimitByObjectType: Map[String, LevelLimit] = Map(
    "sponge-volume"            -> cubeSpongeLevel,
    "sponge-surface"           -> cubeSpongeLevel,
    "tesseract-sponge-volume"  -> tesseractSpongeVolumeLevel,
    "tesseract-sponge-surface" -> tesseractSpongeSurfaceLevel,
  )

  /** `sponge-recursive-ias` accepts levels from `recursiveIasMinLevel` up to (excluding)
    * `recursiveIasMaxLevel + 1`; it has no entry in `levelLimitByObjectType` because it shares
    * none of the mesh-based types' ceilings (usability review 2026-10, session 3, F59). */
  val recursiveIasMinLevel = 1
  val recursiveIasMaxLevel = 13

  /** Recursion-depth ceiling shared by all three GPU-projected 4D IFS types (menger4d,
    * sierpinski4d, hexadecachoron4d, `Instanced4DSceneBuilder`'s `IFS4DType.maxLevel`). Each
    * shader's own traversal stack (`S4D_MAX_STACK`, `H4D_MAX_STACK`) already guards against
    * overflow by silently pruning, so this ceiling is about avoiding a pointlessly slow,
    * silently-truncated-looking render rather than a crash -- same number as menger4d's
    * pre-existing `MAX_4D_LEVEL` bound, applied uniformly instead of only to one of the three.
    */
  val ifs4dMaxLevel = 14

  /** `LSystem.iterations` upper bound (`SceneObject.scala`'s `LSystem` already enforces this
    * at DSL construction time; the CLI `--objects type=lsystem:level=N` path did not). */
  val lsystemMaxIterations = 12

  /** `ParametricSurface.uSteps * vSteps` upper bound -- was a log-and-continue warning
    * (`MemoryWarningThreshold`), now a hard reject. */
  val parametricSurfaceMaxSamples = 1_000_000L
