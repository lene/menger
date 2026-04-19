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
  require(photonsPerIteration > 0 && photonsPerIteration <= OptiXCausticsConfig.MaxPhotonsPerIteration,
    s"photonsPerIteration must be 1-${OptiXCausticsConfig.MaxPhotonsPerIteration}, got $photonsPerIteration")
  require(iterations > 0 && iterations <= OptiXCausticsConfig.MaxIterations,
    s"iterations must be 1-${OptiXCausticsConfig.MaxIterations}, got $iterations")
  require(initialRadius > 0.0f && initialRadius <= OptiXCausticsConfig.MaxInitialRadius,
    s"initialRadius must be 0.0-${OptiXCausticsConfig.MaxInitialRadius}, got $initialRadius")
  require(alpha > 0.0f && alpha < 1.0f,
    s"alpha must be 0.0-1.0 exclusive, got $alpha")

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
