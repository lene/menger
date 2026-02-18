package menger.dsl

import menger.optix.{Material => OptixMaterial}

/** Material definition for DSL with presets and factory methods */
case class Material(
  color: Color = Color.White,
  ior: Float = 1.0f,
  roughness: Float = 0.5f,
  metallic: Float = 0.0f,
  specular: Float = 0.5f,
  emission: Float = 0.0f
):
  require(ior >= 0f, s"IOR must be non-negative, got $ior")
  require(roughness >= 0f && roughness <= 1f, s"Roughness must be in [0, 1], got $roughness")
  require(metallic >= 0f && metallic <= 1f, s"Metallic must be in [0, 1], got $metallic")
  require(specular >= 0f && specular <= 1f, s"Specular must be in [0, 1], got $specular")
  require(emission >= 0f, s"Emission must be non-negative, got $emission")

  def toOptixMaterial: OptixMaterial =
    OptixMaterial(
      color = color.toCommonColor,
      ior = ior,
      roughness = roughness,
      metallic = metallic,
      specular = specular,
      emission = emission
    )

object Material:
  private def fromOptix(m: OptixMaterial): Material =
    Material(Color.fromCommon(m.color), m.ior, m.roughness, m.metallic, m.specular, m.emission)

  // Dielectric presets — delegate to OptixMaterial to avoid duplication
  val Glass   = fromOptix(OptixMaterial.Glass)
  val Water   = fromOptix(OptixMaterial.Water)
  val Diamond = fromOptix(OptixMaterial.Diamond)

  // Metal presets — delegate to OptixMaterial
  val Chrome = fromOptix(OptixMaterial.Chrome)
  val Gold   = fromOptix(OptixMaterial.Gold)
  val Copper = fromOptix(OptixMaterial.Copper)

  // Semi-transparent presets — delegate to OptixMaterial
  val Film      = fromOptix(OptixMaterial.Film)
  val Parchment = fromOptix(OptixMaterial.Parchment)

  // Opaque presets (no equivalent in OptixMaterial named presets)
  val Plastic = Material(Color.White, ior = 1.5f, roughness = 0.3f, metallic = 0f, specular = 0.5f)
  val Matte   = Material(Color.White, ior = 1.0f, roughness = 1.0f, metallic = 0f, specular = 0f)

  // Factory methods — delegate to OptixMaterial
  def matte(color: Color): Material   = fromOptix(OptixMaterial.matte(color.toCommonColor))
  def plastic(color: Color): Material = fromOptix(OptixMaterial.plastic(color.toCommonColor))
  def metal(color: Color): Material   = fromOptix(OptixMaterial.metal(color.toCommonColor))
  def glass(color: Color): Material   = fromOptix(OptixMaterial.glass(color.toCommonColor))
