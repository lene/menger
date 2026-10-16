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

  // One limit per 3D sponge type: they draw the same sponge at very different costs (usability
  // review 2026-10, session 4, F86). `Const.Engine.cubeSpongeMaxLevel` predates the split and
  // names the 3D ("cube-based") sponges as a family, not the `cube-sponge` type.

  /** `sponge-surface`: a mesh of the visible surface only. */
  val surfaceSpongeLevel: LevelLimit =
    LevelLimit(Const.Engine.spongeLevelWarningThreshold, Const.Engine.cubeSpongeMaxLevel)

  /** `sponge-volume`: one CPU mesh of the 20^level cubes' visible faces. Level 5 (13.1M
    * triangles) used to run the render window out of memory while it was merged level by
    * level; built into one array it opens in about 10 s at about 4.5 GB (measured 2026-10-13). */
  val volumeSpongeLevel: LevelLimit =
    LevelLimit(Const.Engine.spongeLevelWarningThreshold, Const.Engine.cubeSpongeMaxLevel)

  /** `cube-sponge` (deprecated, menger#76): one GPU instance per cube, so the instance table is
    * the ceiling. Level 3 is 8000 instances and level 2.x at most 10800
    * (`CubeSpongeGenerator.cubeCount`), but any level above 3 needs 216000, more than
    * `Const.maxInstancesLimit`. It had no limit of its own, and the manifest claimed 5. */
  val cubeSpongeInstancedMaxLevel = 3
  val cubeSpongeInstancedLevel: LevelLimit =
    LevelLimit(Const.Engine.spongeLevelWarningThreshold, cubeSpongeInstancedMaxLevel)
  val tesseractSpongeVolumeLevel: LevelLimit =
    LevelLimit(Const.Engine.tesseractSpongeWarnLevel, Const.Engine.tesseractSpongeMaxLevel)
  val tesseractSpongeSurfaceLevel: LevelLimit =
    LevelLimit(Const.Engine.tesseractSponge2WarnLevel, Const.Engine.tesseractSponge2MaxLevel)

  /** Keyed by normalized object-type name (`menger.common.ObjectType.normalize`). Shared by
    * `InteractiveEngine`'s level table and `ObjectSpec.validateSpongeLevel`. */
  val levelLimitByObjectType: Map[String, LevelLimit] = Map(
    "sponge-volume"            -> volumeSpongeLevel,
    "sponge-surface"           -> surfaceSpongeLevel,
    "cube-sponge"              -> cubeSpongeInstancedLevel,
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

  /** `RenderSettings.accumulation` / `--accumulation-frames` upper bound. Every accumulation frame
    * re-runs the whole render, caustics included; the corpus goes up to 64. */
  val renderAccumulationMax = 256

  // --- Scene-wide budget (review 2026-10-14, menger#80) -----------------------------------
  // Each limit above caps one field; the budget caps what they multiply into, so a scene with
  // every field inside its own cap still cannot hang or run the host renderer out of memory.
  // Checked by `SceneCost` in SceneValidator (a scene defect the agent can repair) and before
  // every engine build.

  /** Triangles held at once by all mesh objects, both levels of a fractional one counted. One
    * level-5 volume sponge (13.1M, about 4.5 GB heap) fits; two -- which ran the render window
    * out of memory in usability sessions 3 and 4 -- do not. */
  val sceneMaxTriangles = 16_000_000L

  /** Whole renders per displayed frame: level-mix worlds (`LevelMix.worlds`) x accumulation. */
  val sceneMaxRendersPerFrame = 128L

  /** Caustics photons traced per displayed frame: photons x iterations x renders per frame.
    * `Caustics.HighQuality` (10^7 per render) at 128 renders fits. */
  val sceneMaxPhotonsPerFrame = 2_000_000_000L
