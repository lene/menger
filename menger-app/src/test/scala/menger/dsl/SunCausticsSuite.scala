package menger.dsl

import io.github.lene.optix.CausticsStats
import menger.common.ImageSize
import menger.common.ProfilingConfig
import menger.common.Vector
import menger.engines.scene.GPURequired
import menger.geometry.MengerRenderer
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** A `Sun` as the scene's first light runs through the photon pass: it emits photons, a caustic
  * is deposited, and the deposited energy matches the same scene lit by a `Directional` of the
  * same direction and intensity (menger#158). GPU-required: cancels without an OptiX device. */
class SunCausticsSuite extends AnyFlatSpec with Matchers:

  private given ProfilingConfig = ProfilingConfig.disabled

  private val Size = ImageSize(200, 150)
  // Grid-aligned (48 * 1024), see CausticsStatsSuite.
  private val Photons = 49152
  private val Iterations = 4
  private val InitialRadius = 1.0f
  private val RadiusAlpha = 0.7f
  private val FloorY = -2f
  private val FloorGrey = 0.8f
  private val GlassIor = 1.5f
  private val Direction = Vec3(0.3f, -1f, 0.2f)
  // The sun's disk is a cone, a Directional a single ray; measured 1.5 % apart, the rest of the
  // 5 % is photon sampling noise.
  private val SameEnergy = 0.05

  private def causticStats(light: Light): CausticsStats =
    val renderer =
      try
        val r = MengerRenderer()
        r.initialize()
        r
      catch case _: Throwable => cancel("OptiX native library not available")
    try
      renderer.setCamera(
        Vector[3](0f, 1f, 4f), Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f), 45f
      )
      renderer.setSphere(Vector[3](0f, 0f, 0f), 1f)
      renderer.setIOR(GlassIor)
      renderer.addPlaneSolidColor(1, positive = false, FloorY, FloorGrey, FloorGrey, FloorGrey)
      renderer.setLights(Array(light.toCommonLight))
      renderer.setShadows(true)
      renderer.enableCaustics(Photons, Iterations, InitialRadius, RadiusAlpha)
      val _ = renderer.render(Size)
      Option(renderer.getCausticsStats).getOrElse(fail("no caustics stats returned"))
    finally renderer.dispose()

  "A Sun as the first light" should "emit photons and deposit a caustic" taggedAs GPURequired in:
    val stats = causticStats(Sun(Direction))
    info(s"photons emitted ${stats.photonsEmitted}, deposited ${stats.photonsDeposited}, " +
      s"flux deposited ${stats.totalFluxDeposited}")
    stats.photonsEmitted should be > 0L
    stats.photonsDeposited should be > 0L
    stats.totalFluxDeposited should be > 0.0

  it should "deposit the energy of a Directional of the same direction and intensity" taggedAs
    GPURequired in:
    val sun = causticStats(Sun(Direction, intensity = 1f))
    val directional = causticStats(Directional(Direction, intensity = 1f))
    info(s"flux emitted ${sun.totalFluxEmitted} / ${directional.totalFluxEmitted}, " +
      s"deposited ${sun.totalFluxDeposited} / ${directional.totalFluxDeposited} (Sun / Directional)")
    directional.totalFluxDeposited should be > 0.0
    math.abs(sun.totalFluxDeposited - directional.totalFluxDeposited) /
      directional.totalFluxDeposited should be < SameEnergy
