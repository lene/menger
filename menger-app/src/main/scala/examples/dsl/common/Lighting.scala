package examples.dsl.common

import scala.language.implicitConversions

import menger.dsl._

/**
 * Reusable lighting setup definitions
 *
 * This object provides pre-configured lighting setups for common
 * rendering scenarios. These can be imported and used as-is, or
 * modified for specific needs.
 *
 * Usage in scenes:
 * {{{
 *   import examples.dsl.common.Lighting._
 *
 *   Scene(
 *     lights = ThreePointLighting,
 *     objects = List(...)
 *   )
 * }}}
 */
object Lighting:
  /**
   * Classic three-point lighting setup
   *
   * Standard cinematography lighting with:
   * - Key light: Main illumination from upper right
   * - Fill light: Softer light from left to reduce shadows
   * - Back light: Rim light from behind for edge definition
   *
   * Good for: General purpose rendering, product visualization
   */
  val ThreePointLighting = List(
    // Key light (main)
    Directional(
      direction = (1f, -1f, -1f),
      intensity = 1.5f
    ),
    // Fill light (soften shadows)
    Directional(
      direction = (-1f, -0.5f, -1f),
      intensity = 0.5f
    ),
    // Back light (rim lighting)
    Directional(
      direction = (0f, 0.5f, 1f),
      intensity = 0.8f
    )
  )

  /**
   * Dramatic single-light setup
   *
   * Strong directional light with deep shadows for
   * high-contrast, dramatic rendering.
   *
   * Good for: Artistic renders, emphasizing form
   */
  val DramaticLighting = List(
    Directional(
      direction = (1f, -1f, -0.5f),
      intensity = 2.0f
    )
  )

  /**
   * Soft ambient-style lighting
   *
   * Multiple soft lights from different angles to create
   * even, shadow-free illumination.
   *
   * Good for: Material showcases, technical visualization
   */
  val SoftAmbientLighting = List(
    Directional(
      direction = (1f, -1f, -1f),
      intensity = 0.8f
    ),
    Directional(
      direction = (-1f, -1f, -1f),
      intensity = 0.8f
    ),
    Directional(
      direction = (0f, 1f, 0f),
      intensity = 0.6f
    )
  )

  /**
   * Golden hour lighting
   *
   * Warm-toned lighting simulating sunset/sunrise conditions.
   * Uses warm key light and cool fill for color contrast.
   *
   * Good for: Atmospheric renders, outdoor scenes
   */
  val GoldenHourLighting = List(
    // Warm key light (sun)
    Directional(
      direction = (1f, -0.3f, -1f),
      intensity = 1.8f,
      color = "#FFCC88"  // Warm orange
    ),
    // Cool fill light (sky)
    Directional(
      direction = (-1f, 0.5f, 1f),
      intensity = 0.4f,
      color = "#88AAFF"  // Cool blue
    )
  )

  /**
   * Studio lighting with point lights
   *
   * Multiple point lights positioned like a photography studio
   * with softboxes.
   *
   * Good for: Product photography, clean professional look
   */
  val StudioLighting = List(
    // Main light - upper right
    Point(
      position = (3f, 4f, 3f),
      intensity = 2.0f
    ),
    // Fill light - upper left
    Point(
      position = (-3f, 4f, 3f),
      intensity = 1.0f
    ),
    // Back light - behind and above
    Point(
      position = (0f, 5f, -3f),
      intensity = 1.5f
    )
  )

  /**
   * Rim lighting setup
   *
   * Emphasizes edges and silhouettes with back lighting.
   *
   * Good for: Dramatic silhouettes, highlighting form
   */
  val RimLighting = List(
    // Front key (low intensity)
    Directional(
      direction = (0f, -0.5f, -1f),
      intensity = 0.5f
    ),
    // Strong rim from behind
    Directional(
      direction = (0f, 0f, 1f),
      intensity = 2.0f
    )
  )

  /**
   * Colored accent lighting
   *
   * Multiple colored lights for artistic/stylized rendering.
   * Cyan and magenta create a modern, vibrant look.
   *
   * Good for: Artistic renders, modern aesthetics
   */
  val ColoredAccentLighting = List(
    // Cyan from left
    Point(
      position = (-4f, 3f, 2f),
      intensity = 1.5f,
      color = "#00FFFF"
    ),
    // Magenta from right
    Point(
      position = (4f, 3f, 2f),
      intensity = 1.5f,
      color = "#FF00FF"
    ),
    // White key from front
    Directional(
      direction = (0f, -1f, -1f),
      intensity = 0.8f
    )
  )

  /**
   * Night scene lighting
   *
   * Cool-toned, low-intensity lighting simulating moonlight
   * and artificial light sources.
   *
   * Good for: Night scenes, moody atmosphere
   */
  val NightSceneLighting = List(
    // Moonlight (cool blue, low angle)
    Directional(
      direction = (1f, -0.3f, -1f),
      intensity = 0.6f,
      color = "#AACCFF"
    ),
    // Artificial light (warm accent)
    Point(
      position = (-2f, 2f, 2f),
      intensity = 0.8f,
      color = "#FFCC88"
    )
  )
