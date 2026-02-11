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
  // Dielectric presets (transparent materials with refraction)
  val Glass = Material(
    color = Color(1f, 1f, 1f, 0.02f),
    ior = 1.5f,
    roughness = 0f,
    metallic = 0f,
    specular = 1f
  )

  val Water = Material(
    color = Color(1f, 1f, 1f, 0.02f),
    ior = 1.33f,
    roughness = 0f,
    metallic = 0f,
    specular = 1f
  )

  val Diamond = Material(
    color = Color(1f, 1f, 1f, 0.02f),
    ior = 2.42f,
    roughness = 0f,
    metallic = 0f,
    specular = 1f
  )

  // Metal presets (colored reflections, no refraction)
  val Chrome = Material(
    color = Color(0.9f, 0.9f, 0.9f),
    ior = 1f,
    roughness = 0f,
    metallic = 1f,
    specular = 1f
  )

  val Gold = Material(
    color = Color(1f, 0.84f, 0f),
    ior = 1f,
    roughness = 0.1f,
    metallic = 1f,
    specular = 1f
  )

  val Copper = Material(
    color = Color(0.72f, 0.45f, 0.20f),
    ior = 1f,
    roughness = 0.2f,
    metallic = 1f,
    specular = 1f
  )

  // Semi-transparent materials
  val Film = Material(
    color = Color(1f, 1f, 1f, 0.2f),
    ior = 1.1f,
    roughness = 0.1f,
    metallic = 0f,
    specular = 0.5f
  )

  val Parchment = Material(
    color = Color(245f/255f, 222f/255f, 179f/255f, 0.4f),
    ior = 1.0f,
    roughness = 0.5f,
    metallic = 0f,
    specular = 0.2f
  )

  // Opaque presets for convenience
  val Plastic = Material(
    color = Color.White,
    ior = 1.5f,
    roughness = 0.3f,
    metallic = 0f,
    specular = 0.5f
  )

  val Matte = Material(
    color = Color.White,
    ior = 1.0f,
    roughness = 1.0f,
    metallic = 0f,
    specular = 0f
  )

  // Factory methods for custom colors
  def matte(color: Color): Material =
    Material(color, ior = 1f, roughness = 1f, metallic = 0f, specular = 0f)

  def plastic(color: Color): Material =
    Material(color, ior = 1.5f, roughness = 0.3f, metallic = 0f, specular = 0.5f)

  def metal(color: Color): Material =
    Material(color, ior = 1f, roughness = 0.1f, metallic = 1f, specular = 1f)

  def glass(color: Color): Material =
    Material(color.copy(a = 0.02f), ior = 1.5f, roughness = 0f, metallic = 0f, specular = 1f)
