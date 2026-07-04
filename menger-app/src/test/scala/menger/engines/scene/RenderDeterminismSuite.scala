package menger.engines.scene

import io.github.lene.optix.MengerRenderer
import menger.ObjectSpec
import menger.common.Color
import menger.common.ImageSize
import menger.common.ProfilingConfig
import menger.common.Vector
import menger.common.{Light => CommonLight}
import menger.config.SceneConfig
import org.scalatest.Tag
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object GPURequired extends Tag("menger.tags.GPURequired")

class RenderDeterminismSuite extends AnyFlatSpec with Matchers:

  private def spec(s: String): ObjectSpec = ObjectSpec.parse(s).toOption.get

  // ── Scene configuration determinism ────────────────────────────────────────

  "SceneConfig" should "be deterministic for identical specs" in:
    val specs1 = List(ObjectSpec("sphere"), ObjectSpec("cube"))
    val specs2 = List(ObjectSpec("sphere"), ObjectSpec("cube"))
    val config1 = SceneConfig.multiObject(specs1)
    val config2 = SceneConfig.multiObject(specs2)
    config1 shouldBe config2

  it should "be deterministic for identical 4D projection specs" in:
    val s1 = spec("type=tesseract:level=2:rot-xw=30:rot-yw=15:eye-w=4:screen-w=2")
    val s2 = spec("type=tesseract:level=2:rot-xw=30:rot-yw=15:eye-w=4:screen-w=2")
    val config1 = SceneConfig.multiObject(List(s1))
    val config2 = SceneConfig.multiObject(List(s2))
    config1 shouldBe config2

  it should "differ for objects with different positions" in:
    val s1 = spec("type=sphere:pos=0,0,0")
    val s2 = spec("type=sphere:pos=1,0,0")
    val config1 = SceneConfig.multiObject(List(s1))
    val config2 = SceneConfig.multiObject(List(s2))
    config1 should not be config2

  // ── GPU render determinism ──────────────────────────────────────────────────

  private given ProfilingConfig = ProfilingConfig.disabled

  /** Opaque sphere over a solid floor, single hard-shadow sample, antialiasing off.
    * All RNG-driven paths (adaptive AA, accumulation, caustic float-atomic gather) are
    * excluded so the primary-ray trace is bitwise reproducible on a fixed GPU. */
  private def setupDeterministicScene(r: MengerRenderer): Unit =
    r.setAntialiasing(false, 1, 0.1f)
    r.setCamera(
      Vector[3](0.0f, 1.0f, 4.0f),
      Vector[3](0.0f, 0.0f, 0.0f),
      Vector[3](0.0f, 1.0f, 0.0f),
      45.0f
    )
    r.setSphere(Vector[3](0.0f, 0.0f, 0.0f), 1.0f)
    r.addPlaneSolidColor(1, positive = false, -2.0f, 0.8f, 0.8f, 0.8f)
    r.setLights(
      Array(CommonLight.Point(Vector[3](0.0f, 10.0f, 0.0f), Color(1.0f, 1.0f, 1.0f), 500.0f))
    )
    r.setShadows(true)

  // Caustics are deliberately NOT exercised here: the caustic flux gather uses float
  // atomicAdd across photon threads, whose summation order is nondeterministic, so caustic
  // pixels may differ by ~1 LSB between runs. Opaque/refractive primary-ray rendering has no
  // such race and is bitwise reproducible — this test backs the arc42 §10 reproducibility
  // claim. See docs/BACKLOG.md F-CAUSTICS-STATS / RenderDeterminism notes for the caustic case.
  "Render pipeline" should "produce byte-identical output for an identical scene" taggedAs
    GPURequired in:
    val size = ImageSize(160, 120)
    val r =
      try
        val rr = MengerRenderer()
        rr.initialize()
        rr
      catch case _: Throwable => cancel("OptiX native library not available")
    try
      setupDeterministicScene(r)
      val first = r.render(size)
      val second = r.render(size)
      first.length should be > 0
      second shouldBe first
    finally r.dispose()
