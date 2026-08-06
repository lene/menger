package menger.config

import menger.common.Color

/**
 * Material configuration for surface appearance.
 *
 * @param color surface color/albedo
 * @param ior index of refraction (1.0 = no refraction, 1.5 = glass, 2.42 = diamond)
 */
case class MaterialConfig(
  color: Color = Color(1f, 1f, 1f),
  ior: Float = 1.5f
)

object MaterialConfig:
  val Default: MaterialConfig = MaterialConfig()

  val Glass: MaterialConfig = MaterialConfig(
    color = Color(1f, 1f, 1f, 0.1f),
    ior = 1.5f
  )

  val Diamond: MaterialConfig = MaterialConfig(
    color = Color(1f, 1f, 1f),
    ior = 2.42f
  )

  val Mirror: MaterialConfig = MaterialConfig(
    color = Color(1f, 1f, 1f),
    ior = 1.0f
  )

  val Water: MaterialConfig = MaterialConfig(
    color = Color(0.8f, 0.9f, 1.0f, 0.3f),
    ior = 1.33f
  )
