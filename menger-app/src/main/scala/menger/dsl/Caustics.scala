package menger.dsl

import menger.common.{CausticsConfig => OptiXCausticsConfig}

/** Caustics rendering configuration for photon mapping effects.
  *
  * @param enabled Whether caustics rendering is enabled
  * @param photonsPerIteration Number of photons to trace per iteration
  * @param iterations Number of photon tracing iterations
  * @param initialRadius Initial photon-gather radius; `None` auto-derives it from scene
  *                      geometry (optix-jni >= 0.1.13), `Some(r)` is an explicit override
  * @param alpha Radius reduction factor between iterations (0.0-1.0)
  */
case class Caustics(
  enabled: Boolean = true,
  photonsPerIteration: Int = 100000,
  iterations: Int = 10,
  initialRadius: Option[Float] = None,
  alpha: Float = 0.7f
):
  // Delegate validation to CausticsConfig — single source of constraint definitions.
  toCausticsConfig

  def toCausticsConfig: OptiXCausticsConfig =
    OptiXCausticsConfig(
      enabled, photonsPerIteration, iterations,
      initialRadius.getOrElse(OptiXCausticsConfig.AutoRadius), alpha)

object Caustics:
  /** Disabled caustics (no photon mapping) */
  val Disabled: Caustics = Caustics(enabled = false)

  /** Default caustics configuration (explicit gather radius; bare `Caustics()` auto-derives) */
  val Default: Caustics = Caustics(initialRadius = Some(1.0f))

  /** High quality caustics with more photons and iterations */
  val HighQuality: Caustics = Caustics(
    photonsPerIteration = 500000,
    iterations = 20,
    initialRadius = Some(1.0f),
    alpha = 0.8f
  )
