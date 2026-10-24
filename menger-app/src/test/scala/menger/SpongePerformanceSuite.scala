package menger

import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.OptiXRenderer
import io.github.lene.qa.BenchConfig
import io.github.lene.qa.Perf
import io.github.lene.qa.PerfGate
import io.github.lene.qa.RelativeBenchmark
import io.github.lene.qa.Side
import menger.common.Color
import menger.common.ImageSize
import menger.common.ProfilingConfig
import menger.common.TriangleMeshData
import menger.common.Vector
import menger.objects.SpongeBySurface
import menger.objects.SpongeByVolume
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.Tag
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object Slow extends Tag("Slow")

/** Sponge generation and rendering gates, each timed against a reference in interleaved rounds
  * (io.github.lene.qa.RelativeBenchmark). Mesh generation runs on the CPU: one gate per subject
  * against a fixed CPU workload (uniform slowdowns) and one against the same generator a level
  * lower (scaling regressions). Rendering compares against the level-0 cube through the same
  * mesh path.
  * Triangle counts are asserted by SpongeBySurfaceMeshSuite and SpongeByVolumeMeshSuite. */
class SpongePerformanceSuite extends AnyFlatSpec
    with Matchers
    with LazyLogging
    with PerfGate
    with BeforeAndAfterEach:

  given ProfilingConfig = ProfilingConfig.disabled

  private val RenderSize = ImageSize(800, 600)

  // Generation limits (menger#45): thread CPU time with a fixed, pre-touched heap
  // (BenchConfig.JvmCpu, build.sbt). 1.5x the highest upper confidence bound -- not the usual
  // ~2x, which can never fail a 2x slowdown (decided by user 2026-10-03: "1.5x, catch most").
  // Probe gates catch a uniform slowdown, scaling gates (vs the level below) a level-dependent
  // one. Measured over 5 idle runs and 5 with all cores busy plus a 100% GPU burn (2026-10-03).
  // Highest bound -> limit:
  private val MaxSlowdownSurfaceL2 = 0.36           // 0.239
  private val MaxSlowdownSurfaceL3 = 26.0           // 17.27
  private val MaxSlowdownVolumeL2 = 0.69            // 0.459 (loaded)
  private val MaxScalingSurfaceL1ToL2 = 33.4        // 22.28
  private val MaxScalingSurfaceL2ToL3 = 117.0       // 78.13
  private val MaxScalingVolumeL1ToL2 = 34.1         // 22.72
  // Render limits: ~2x the highest upper confidence bound measured on the RTX A1000 laptop
  // (also the CI runner) over 3 idle runs and 5 runs under a 99% GPU burn plus CPU load
  // (2026-09-23). Highest bound -> limit:
  private val MaxSlowdownRenderSurfaceL2 = 2.3      // 1.13 (loaded)
  private val MaxSlowdownRenderVolumeL2 = 2.4       // 1.16
  private val MaxSlowdownRenderTransparentL1 = 3.9  // 1.91

  // A fixed, deterministic CPU workload (allocate and sort) to time mesh generation against.
  private val ProbeSize = 100_000
  private val probeData: Array[Double] = Array.tabulate(ProbeSize)(i => math.sin(i.toDouble))
  private val cpuProbe = Side("CPU probe", () => { val _ = probeData.sorted })

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[OptiXRenderer] = None

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  protected def renderer: OptiXRenderer = rendererOpt.getOrElse(
    throw new IllegalStateException("Renderer not initialized")
  )

  override def withFixture(test: NoArgTest): Outcome =
    if rendererOpt.isDefined then super.withFixture(test)
    else cancel("OptiX native library not available")

  override def beforeEach(): Unit =
    super.beforeEach()
    try
      // Touch the companion first so liboptixjni is actually loaded before any @native call;
      // otherwise the suite depends on another suite having loaded it (run alone, it cancels).
      val _ = OptiXRenderer.isLibraryLoaded
      val r = new OptiXRenderer()
      r.initialize()
      rendererOpt = Some(r)
      setupDefaults()
    catch case _: Throwable => ()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally super.afterEach()
    rendererOpt = None

  protected def setupDefaults(): Unit =
    renderer.setCamera(
      Vector[3](0.0f, 0.5f, 3.0f),
      Vector[3](0.0f, 0.0f, 0.0f),
      Vector[3](0.0f, 1.0f, 0.0f),
      60.0f
    )
    renderer.setLight(Vector[3](0.5f, 0.5f, -0.5f), 1.0f)
    renderer.setSphere(Vector[3](0.0f, 0.0f, 0.0f), 0.5f)

  private def surface(level: Float): TriangleMeshData =
    SpongeBySurface(center = Vector.Zero[3], scale = 2.0f, level = level).toTriangleMesh

  private def volume(level: Float): TriangleMeshData =
    SpongeByVolume(center = Vector.Zero[3], scale = 2.0f, level = level).toTriangleMesh

  private def generation(name: String, generate: => TriangleMeshData): Side =
    Side(name, () => { val _ = generate })

  // IAS instance path (menger#46: the single-mesh clearTriangleMesh/setTriangleMeshColor/
  // setTriangleMeshIOR trio is deprecated since 0.1.5 in favor of instances + Material).
  // Each render side replaces the mesh: addTriangleMesh appends, so clear first.
  private def rendering(name: String, mesh: TriangleMeshData, color: Color, ior: Float): Side =
    Side(
      name,
      () => { val _ = renderer.render(RenderSize) },
      prepare = () =>
        renderer.clearAllInstances()
        renderer.addTriangleMesh(mesh)
        renderer.addTriangleMeshInstance(Vector[3](0f, 0f, 0f), color, ior)
        renderer.clearPlanes()
        renderer.addPlane(1, true, -2.0f)
    )

  private def gate(
      reference: Side,
      subject: Side,
      maxSlowdown: Double,
      config: BenchConfig = BenchConfig()
  ) =
    assertWithin(
      s"${subject.name} vs ${reference.name}",
      RelativeBenchmark.compare(reference, subject, maxSlowdown, config)
    )

  private def generationGate(name: String, generate: => TriangleMeshData, maxSlowdown: Double) =
    gate(cpuProbe, generation(name, generate), maxSlowdown, BenchConfig.JvmCpu)

  private def scalingGate(
      name: String,
      level: Int,
      generate: Float => TriangleMeshData,
      maxRatio: Double
  ) =
    gate(
      generation(s"level ${level - 1} $name", generate(level - 1f)),
      generation(s"level $level $name", generate(level.toFloat)),
      maxRatio,
      BenchConfig.JvmCpu
    )

  "Sponge generation" should "generate a level 2 surface sponge within its limit" taggedAs Perf in:
    generationGate("level 2 surface sponge", surface(2f), MaxSlowdownSurfaceL2)

  it should "generate a level 3 surface sponge within its limit" taggedAs Perf in:
    generationGate("level 3 surface sponge", surface(3f), MaxSlowdownSurfaceL3)

  it should "generate a level 2 volume sponge within its limit" taggedAs Perf in:
    generationGate("level 2 volume sponge", volume(2f), MaxSlowdownVolumeL2)

  it should "scale a surface sponge from level 1 to 2 within its limit" taggedAs Perf in:
    scalingGate("surface sponge", 2, surface, MaxScalingSurfaceL1ToL2)

  it should "scale a surface sponge from level 2 to 3 within its limit" taggedAs Perf in:
    scalingGate("surface sponge", 3, surface, MaxScalingSurfaceL2ToL3)

  it should "scale a volume sponge from level 1 to 2 within its limit" taggedAs Perf in:
    scalingGate("volume sponge", 2, volume, MaxScalingVolumeL1ToL2)

  "Sponge rendering" should "render a level 2 surface sponge within its limit" taggedAs Perf in:
    val grey = Color(0.8f, 0.8f, 0.8f)
    gate(
      rendering("level 0 cube render", surface(0f), grey, 1.0f),
      rendering("level 2 surface sponge render", surface(2f), grey, 1.0f),
      MaxSlowdownRenderSurfaceL2
    )

  it should "render a level 2 volume sponge within its limit" taggedAs Perf in:
    val grey = Color(0.8f, 0.8f, 0.8f)
    gate(
      rendering("level 0 cube render", volume(0f), grey, 1.0f),
      rendering("level 2 volume sponge render", volume(2f), grey, 1.0f),
      MaxSlowdownRenderVolumeL2
    )

  it should "render a transparent level 1 surface sponge within limit" taggedAs Perf in:
    val glass = Color(0.9f, 0.9f, 1.0f, 0.5f)
    gate(
      rendering("transparent level 0 cube render", surface(0f), glass, 1.5f),
      rendering("transparent level 1 surface sponge render", surface(1f), glass, 1.5f),
      MaxSlowdownRenderTransparentL1
    )
