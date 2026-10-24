package menger.dsl

import menger.common.CausticsConfig
import menger.common.RenderLimits

/** Every bound a DSL field has, declared once (review 2026-10-14, menger#114, msa#28): the DSL
  * constructors' `require`s check against it and `ManifestGenerator` exports it, so the agent
  * sees every cap the renderer enforces. Before, the bounds lived in `require`s and again, by
  * hand, in string-keyed tables in `DslSemantics` -- and caps such as `IBL.samples` (1-8) or
  * `RenderSettings.aaMaxDepth` (1-4) were in the `require`s only, invisible to the agent.
  * The numbers themselves stay in `ResourceLimits` and the renderer's own config types. */
object FieldConstraints:

  /** A field's bounds: values outside [min, max] are rejected, `warnAt` and above is slow. */
  final case class Bounds(min: Option[Double], max: Option[Double], warnAt: Option[Double] = None):
    def admits(value: Double): Boolean =
      min.forall(value >= _) && max.forall(value <= _)

  object Bounds:
    def between(min: Double, max: Double): Bounds = Bounds(Some(min), Some(max))
    def level(limit: ResourceLimits.LevelLimit): Bounds =
      Bounds(Some(0d), Some(limit.max.toDouble), Some(limit.warnAt.toDouble))

  /** Bounds that depend on another field's value (`Sponge.level` on `spongeType`). */
  final case class BySubtype(field: String, values: Map[String, Bounds])

  /** A cap on the product of several fields of one type (`uSteps * vSteps`), which no per-field
    * bound can express. */
  final case class ProductCap(typeName: String, fields: List[String], max: Double)

  private val L = ResourceLimits

  /** Per-variant level bounds; the field-level bound below is the conservative minimum of them. */
  val bySubtype: Map[(String, String), BySubtype] = Map(
    ("Sponge", "level") -> BySubtype("spongeType", Map(
      "VolumeFilling"    -> Bounds.level(L.volumeSpongeLevel),
      "SurfaceUnfolding" -> Bounds.level(L.surfaceSpongeLevel),
      "RecursiveIAS"     ->
        Bounds(Some(L.recursiveIasMinLevel.toDouble), Some(L.recursiveIasMaxLevel.toDouble))
    )),
    ("TesseractSponge", "level") -> BySubtype("spongeType", Map(
      "VolumeRemoving"     -> Bounds.level(L.tesseractSpongeVolumeLevel),
      "SurfaceSubdividing" -> Bounds.level(L.tesseractSpongeSurfaceLevel)
    ))
  )

  val fields: Map[(String, String), Bounds] = Map(
    ("Sponge", "level") -> Bounds(Some(0d),
      Some(math.min(L.volumeSpongeLevel.max, L.surfaceSpongeLevel.max).toDouble),
      Some(math.min(L.volumeSpongeLevel.warnAt, L.surfaceSpongeLevel.warnAt).toDouble)),
    ("TesseractSponge", "level") -> Bounds(Some(0d),
      Some(math.min(L.tesseractSpongeVolumeLevel.max, L.tesseractSpongeSurfaceLevel.max).toDouble),
      Some(math.min(L.tesseractSpongeVolumeLevel.warnAt, L.tesseractSpongeSurfaceLevel.warnAt)
        .toDouble)),
    ("PentachoronGasket", "level") -> Bounds.between(0d, L.ifs4dMaxLevel.toDouble),
    ("Menger4D", "level") -> Bounds.between(0d, L.ifs4dMaxLevel.toDouble),
    ("HexadecachoronGasket", "level") -> Bounds.between(0d, L.ifs4dMaxLevel.toDouble),
    ("LSystem", "iterations") -> Bounds.between(0d, L.lsystemMaxIterations.toDouble),
    ("ParametricSurface", "uSteps") -> Bounds.between(1d, L.parametricSurfaceMaxSamples.toDouble),
    ("ParametricSurface", "vSteps") -> Bounds.between(1d, L.parametricSurfaceMaxSamples.toDouble),
    ("RenderSettings", "accumulation") -> Bounds.between(1d, L.renderAccumulationMax.toDouble),
    ("RenderSettings", "aaMaxDepth") -> Bounds.between(1d, 4d),
    ("RenderSettings", "aaThreshold") -> Bounds.between(0d, 1d),
    ("RenderSettings", "maxRayDepth") -> Bounds.between(1d, RenderLimits.MaxRayDepth.toDouble),
    ("IBL", "samples") -> Bounds.between(1d, 8d),
    ("AreaLight", "shadowSamples") -> Bounds.between(1d, 16d),
    ("Caustics", "photonsPerIteration") ->
      Bounds.between(1d, CausticsConfig.MaxPhotonsPerIteration.toDouble),
    ("Caustics", "iterations") -> Bounds.between(1d, CausticsConfig.MaxIterations.toDouble)
  )

  val products: List[ProductCap] = List(
    ProductCap("ParametricSurface", List("uSteps", "vSteps"), L.parametricSurfaceMaxSamples.toDouble)
  )

  /** Rejects `value` if it is outside `typeName.field`'s bounds -- the `require` a constructor
    * calls, with one message format for every field. */
  def check(typeName: String, field: String, value: Double): Unit =
    fields.get((typeName, field)).foreach { b =>
      require(b.admits(value), s"$typeName.$field must be ${describe(b)}, got ${shown(value)}")
    }

  private def describe(b: Bounds): String = (b.min, b.max) match
    case (Some(lo), Some(hi)) => s"${shown(lo)}-${shown(hi)}"
    case (Some(lo), None)     => s">= ${shown(lo)}"
    case (None, Some(hi))     => s"<= ${shown(hi)}"
    case _                    => "any value"

  private def shown(v: Double): String = if v == math.rint(v) then v.toLong.toString else v.toString
