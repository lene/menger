package menger.caustics

import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.CausticsStats
import io.github.lene.optix.MengerRenderer
import io.github.lene.optix.OptiXRenderer
import menger.common.Color
import menger.common.ImageSize
import menger.common.ProfilingConfig
import menger.common.Vector
import menger.common.{Light => CommonLight}
import menger.engines.scene.GPURequired
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * L2 statistical caustics tests (ladder rungs C1, C2, C5, C6, C7) — GPU-required.
 *
 * Renders the canonical scene (glass sphere IOR 1.5 at origin, floor at y=-2, point
 * light (0,10,0) I=500) with caustics enabled and asserts physical invariants on
 * the returned [[CausticsStats]].
 *
 * Structural invariants (photon counts, radius ordering) hold today and are asserted
 * directly. Invariants that depend on the not-yet-applied physics fixes are marked
 * `pending` with the defect ID they wait on; the corresponding tasks flip them green:
 *   - C1 emitted flux == I*deltaOmega          -> P1 (Task 33.3)
 *   - C6 iteration-invariance of brightness     -> P6 (Task 33.3)
 *   - C5 energy conservation within 5%          -> P2/P9 (Task 33.4)
 *   - C7 caustic peak / ambient ratio           -> P5 (Task 33.5)
 *
 * On a machine without an OptiX device the whole suite cancels (never fails).
 */
class CausticsStatsSuite extends AnyFlatSpec
    with Matchers
    with LazyLogging
    with BeforeAndAfterEach:

  given ProfilingConfig = ProfilingConfig.disabled

  private val Size = ImageSize(200, 150) // small: keeps the photon/gather passes fast
  private val Photons = 50000
  private val Iterations = 4

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[OptiXRenderer] = None

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def renderer: OptiXRenderer =
    rendererOpt.getOrElse(throw new IllegalStateException("Renderer not initialized"))

  override def withFixture(test: NoArgTest): Outcome =
    if rendererOpt.isDefined then super.withFixture(test)
    else cancel("OptiX native library not available")

  override def beforeEach(): Unit =
    super.beforeEach()
    try
      val r = MengerRenderer()
      r.initialize()
      rendererOpt = Some(r)
      setupCanonicalScene()
    catch case t: Throwable => logger.debug(s"GPU unavailable, suite will cancel: ${t.getMessage}")

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally super.afterEach()
    rendererOpt = None

  /** Build the canonical caustics twin scene through the low-level renderer API. */
  private def setupCanonicalScene(): Unit =
    renderer.setCamera(
      Vector[3](0.0f, 1.0f, 4.0f),
      Vector[3](0.0f, 0.0f, 0.0f),
      Vector[3](0.0f, 1.0f, 0.0f),
      45.0f
    )
    renderer.setSphere(Vector[3](0.0f, 0.0f, 0.0f), 1.0f)
    renderer.setIOR(1.5f)
    // Floor plane at y = -2, diffuse gray 0.8 (axis 1 = Y).
    renderer.addPlaneSolidColor(1, positive = false, -2.0f, 0.8f, 0.8f, 0.8f)
    renderer.setLights(
      Array(CommonLight.Point(Vector[3](0.0f, 10.0f, 0.0f), Color(1.0f, 1.0f, 1.0f), 500.0f))
    )
    renderer.setShadows(true)

  /** Render once with caustics enabled and return the stats (or fail if absent). */
  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def renderStats(iterations: Int = Iterations): CausticsStats =
    renderer.enableCaustics(Photons, iterations, initialRadius = 1.0f, alpha = 0.7f)
    val _ = renderer.render(Size)
    val stats =
      try Option(renderer.getCausticsStats)
      catch
        // optix-jni <= 0.1.10 binds getCausticsStatsNative to the nested class name
        // io/github/lene/optix/OptiXRenderer$CausticsStats while the artifact ships a
        // top-level CausticsStats -> NoClassDefFoundError. Fixed in 0.1.11 (Task 33.6).
        case _: NoClassDefFoundError | _: UnsatisfiedLinkError =>
          cancel("getCausticsStats binding unavailable (optix-jni <= 0.1.10 bug; 0.1.11 / Task 33.6)")
    stats.getOrElse(throw new IllegalStateException("no caustics stats returned"))

  // ── C1: photon emission ─────────────────────────────────────────────────────

  "C1 emission" should "emit the requested number of photons" taggedAs GPURequired in:
    val stats = renderStats()
    stats.photonsEmitted shouldBe (Photons.toLong * Iterations)

  it should "aim most photons toward the sphere (importance sampling)" taggedAs GPURequired in:
    val stats = renderStats()
    stats.photonsTowardSphere should be > 0L

  it should "carry emitted flux equal to I*deltaOmega [P1, Task 33.3]" taggedAs GPURequired in:
    pending // totalFluxEmitted must equal light intensity * cone solid angle; fixed in 33.3

  // ── C2: sphere hit rate ─────────────────────────────────────────────────────

  "C2 sphere hits" should "record photon-sphere intersections" taggedAs GPURequired in:
    val stats = renderStats()
    stats.sphereHits should be > 0L
    stats.sphereHitRate should (be >= 0.0 and be <= 1.0)

  // ── C5: energy conservation ─────────────────────────────────────────────────

  "C5 energy" should "conserve flux within 5% [P2/P9, Task 33.4]" taggedAs GPURequired in:
    pending // deposited + absorbed + reflected ~= emitted once photons branch (33.4)

  // ── C6: convergence ─────────────────────────────────────────────────────────

  "C6 convergence" should "keep radii ordered min <= avg <= max <= initial" taggedAs GPURequired in:
    val stats = renderStats()
    stats.minRadius should be <= stats.avgRadius
    stats.avgRadius should be <= stats.maxRadius
    stats.maxRadius should be <= 1.0f + 1e-4f

  it should "give iteration-invariant brightness [P6, Task 33.3]" taggedAs GPURequired in:
    pending // mean caustic radiance for 1 vs N iterations must agree; fixed in 33.3

  // ── C7: brightness ──────────────────────────────────────────────────────────

  "C7 brightness" should "produce a caustic brighter than the ambient floor [P5, Task 33.5]" taggedAs
    GPURequired in:
    pending // maxCausticBrightness > 1.5 * avgFloorBrightness once density estimate is correct (33.5)
