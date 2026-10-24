package menger.engines.scene

import io.github.lene.optix.LevelMix
import io.github.lene.optix.MengerRenderer
import io.github.lene.optix.OptiXRenderer
import menger.common.Material

/** Which fractional-level objects are drawn as their two whole levels and mixed by the renderer
  * ([[LevelMix]]), instead of fading level n's hole caps over level n+1 in one render.
  *
  * Refractive materials only: a glass cap shows level n+1's new tunnels behind it at any
  * strength, so the level jumped at the integer (usability review 2026-10, session 4, F85). An
  * opaque cap hides what lies behind it, so fading it already mixes the two levels in one render
  * (measured pixel for pixel, 2026-10-14). Only a [[MengerRenderer]] can mix; with a plain
  * renderer the caps stay. */
object LevelMixing:

  // A face refracts above this IOR -- the threshold the triangle shader (is_refractive) and the
  // caustics photon targets use.
  private val RefractiveIor = 1.05f

  def isRefractive(material: Material): Boolean = material.ior > RefractiveIor

  /** Whether an object at `level` in `material` is drawn as two levels. */
  def appliesTo(level: Option[Float], material: Material): Boolean =
    level.exists(l => l != l.floor) && isRefractive(material)

  /** The renderer to register two-level objects with, if it can mix. */
  def mixer(renderer: OptiXRenderer): Option[MengerRenderer] = renderer match
    case menger: MengerRenderer => Some(menger)
    case _ => None

  /** Registers an object at fractional `level` whose level n is drawn by the instances `lower`
    * and whose level n+1 by `upper`. */
  def register(mixer: MengerRenderer, lower: Seq[Int], upper: Seq[Int], level: Float): Unit =
    mixer.addLevelMix(LevelMix(lower.toVector, upper.toVector, level - level.floor))
