package menger.dsl

import menger.optix.{CausticsConfig => OptiXCausticsConfig}

/** Caustics rendering configuration for photon mapping effects.
  *
  * @param enabled Whether caustics rendering is enabled
  * @param photonsPerIteration Number of photons to trace per iteration
  * @param iterations Number of photon tracing iterations
  * @param initialRadius Initial search radius for photon gathering
  * @param alpha Radius reduction factor between iterations (0.0-1.0)
  */
case class Caustics(
  enabled: Boolean = true,
  photonsPerIteration: Int = 100000,
  iterations: Int = 10,
  initialRadius: Float = 1.0f,
  alpha: Float = 0.7f
):
  // Delegate validation to CausticsConfig — single source of constraint definitions.
  toCausticsConfig

  def toCausticsConfig: OptiXCausticsConfig =
    OptiXCausticsConfig(enabled, photonsPerIteration, iterations, initialRadius, alpha)

object Caustics:
  /** Disabled caustics (no photon mapping) */
  val Disabled: Caustics = Caustics(enabled = false)

  /** Default caustics configuration */
  val Default: Caustics = Caustics()

  /** High quality caustics with more photons and iterations */
  val HighQuality: Caustics = Caustics(
    photonsPerIteration = 500000,
    iterations = 20,
    initialRadius = 1.0f,
    alpha = 0.8f
  )
