package examples.dsl.common

import scala.language.implicitConversions

import menger.dsl.*

/**
 * Reusable custom material definitions
 *
 * This object provides a library of custom materials that can be
 * imported and used across multiple scenes. Materials here extend
 * the built-in presets with custom colors, properties, and combinations.
 *
 * Usage in scenes:
 * {{{
 *   import examples.dsl.common.Materials.*
 *
 *   Scene(
 *     objects = List(
 *       Sphere(TintedGlass),
 *       Cube(BrushedGold)
 *     )
 *   )
 * }}}
 */
object Materials:
  // Glass variations

  /** Blue-tinted glass for underwater or ice effects */
  val TintedGlass = Material.Glass.copy(
    color = Color(0.85f, 0.9f, 1.0f, 0.02f)
  )

  /** Rose-colored glass with warm tint */
  val RoseGlass = Material.Glass.copy(
    color = Color(1.0f, 0.9f, 0.95f, 0.02f)
  )

  /** Emerald glass with green tint */
  val EmeraldGlass = Material.Glass.copy(
    color = Color(0.85f, 1.0f, 0.9f, 0.02f),
    ior = 1.57f  // Closer to real emerald
  )

  // Metal variations

  /** Brushed gold with higher roughness for matte finish */
  val BrushedGold = Material.Gold.copy(roughness = 0.4f)

  /** Polished copper with lower roughness */
  val PolishedCopper = Material.Copper.copy(roughness = 0.05f)

  /** Aged brass with slightly greenish tint */
  val AgedBrass = Material.metal(Color(0.7f, 0.65f, 0.4f)).copy(
    roughness = 0.3f
  )

  /** Rose gold alloy */
  val RoseGold = Material.metal(Color(0.85f, 0.5f, 0.5f)).copy(
    roughness = 0.1f
  )

  /** Titanium with cool gray tone */
  val Titanium = Material.metal(Color(0.7f, 0.7f, 0.75f)).copy(
    roughness = 0.2f
  )

  // Plastic variations

  /** Glossy red plastic */
  val RedPlastic = Material.plastic(Color.Red).copy(
    roughness = 0.2f,
    specular = 0.7f
  )

  /** Matte cyan plastic */
  val CyanPlastic = Material.plastic(Color("#00FFFF")).copy(
    roughness = 0.5f
  )

  /** Translucent white plastic */
  val TranslucentPlastic = Material.plastic(Color.White).copy(
    color = Color(1f, 1f, 1f, 0.3f),
    roughness = 0.3f
  )

  // Matte variations

  /** Deep matte black */
  val MatteBlack = Material.matte(Color.Black).copy(
    roughness = 1.0f
  )

  /** Soft matte white */
  val MatteWhite = Material.matte(Color.White).copy(
    roughness = 0.95f
  )

  /** Terracotta orange */
  val Terracotta = Material.matte(Color("#CC6633")).copy(
    roughness = 0.9f
  )

  // Special materials

  /** Frosted glass - translucent with high roughness */
  val FrostedGlass = Material.Glass.copy(
    roughness = 0.3f,
    color = Color(1f, 1f, 1f, 0.1f)
  )

  /** Obsidian - dark reflective */
  val Obsidian = Material(
    color = Color(0.1f, 0.1f, 0.15f),
    ior = 1.5f,
    roughness = 0.05f,
    metallic = 0.0f,
    specular = 0.9f
  )

  /** Pearl - subtle iridescent white */
  val Pearl = Material(
    color = Color(0.95f, 0.95f, 1.0f),
    ior = 1.55f,
    roughness = 0.15f,
    metallic = 0.0f,
    specular = 0.8f
  )
