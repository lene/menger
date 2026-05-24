package menger.dsl

import menger.common.{Material => CoreMaterial}

/** Material definition for DSL with presets and factory methods */
case class Material(
  color: Color = Color.White,
  ior: Float = 1.0f,
  roughness: Float = 0.5f,
  metallic: Float = 0.0f,
  specular: Float = 0.5f,
  emission: Float = 0.0f,
  filmThickness: Float = 0.0f
):
  require(ior >= 0f, s"IOR must be non-negative, got $ior")
  require(roughness >= 0f && roughness <= 1f, s"Roughness must be in [0, 1], got $roughness")
  require(metallic >= 0f && metallic <= 1f, s"Metallic must be in [0, 1], got $metallic")
  require(specular >= 0f && specular <= 1f, s"Specular must be in [0, 1], got $specular")
  require(emission >= 0f, s"Emission must be non-negative, got $emission")
  require(filmThickness >= 0f, s"Film thickness must be non-negative, got $filmThickness")

  /** Check material for physical plausibility. Returns advisory warning strings.
    * Never throws — use the existing require() calls for hard failures. */
  def validate(): Seq[String] =
    Seq(
      Option.when(ior < 1.0f)(s"IOR $ior is below 1.0 — unphysical except for metamaterials"),
      Option.when(metallic > 0.0f && ior > 1.1f)(s"Metallic materials typically use IOR=1.0 in PBR, got IOR=$ior"),
      Option.when(metallic > 0.0f && emission > 0.0f)(s"Combining metallic=$metallic with emission=$emission is unusual"),
      Option.when(roughness > 0.9f && metallic > 0.5f)(s"High roughness ($roughness) on metallic material may appear muddy"),
      Option.when(filmThickness > 0.0f && metallic > 0.5f)("Thin-film on metallic surface — visual effect may be minimal")
    ).flatten

  def toCoreMaterial: CoreMaterial =
    CoreMaterial(
      color = color.toCommonColor,
      ior = ior,
      roughness = roughness,
      metallic = metallic,
      specular = specular,
      emission = emission,
      filmThickness = filmThickness
    )

object Material:
  private def fromOptix(m: CoreMaterial): Material =
    Material(Color.fromCommon(m.color), m.ior, m.roughness, m.metallic, m.specular, m.emission, m.filmThickness)

  // Dielectric presets — delegate to CoreMaterial to avoid duplication
  val Glass   = fromOptix(CoreMaterial.Glass)
  val Water   = fromOptix(CoreMaterial.Water)
  val Diamond = fromOptix(CoreMaterial.Diamond)

  // Metal presets — delegate to CoreMaterial
  val Chrome = fromOptix(CoreMaterial.Chrome)
  val Gold   = fromOptix(CoreMaterial.Gold)
  val Copper = fromOptix(CoreMaterial.Copper)

  // Semi-transparent presets — delegate to CoreMaterial
  val Film      = fromOptix(CoreMaterial.Film)
  val Parchment = fromOptix(CoreMaterial.Parchment)

  // Opaque presets — delegate to CoreMaterial to maintain single source of truth
  val Plastic = fromOptix(CoreMaterial.Plastic)
  val Matte   = fromOptix(CoreMaterial.Matte)

  // Factory methods — delegate to CoreMaterial
  def matte(color: Color): Material   = fromOptix(CoreMaterial.matte(color.toCommonColor))
  def plastic(color: Color): Material = fromOptix(CoreMaterial.plastic(color.toCommonColor))
  def metal(color: Color): Material   = fromOptix(CoreMaterial.metal(color.toCommonColor))
  def glass(color: Color): Material   = fromOptix(CoreMaterial.glass(color.toCommonColor))
