package io.github.lene.optix

import java.util.concurrent.atomic.AtomicInteger

import menger.common.Vector
import com.typesafe.scalalogging.LazyLogging
import io.github.lene.qa.BenchConfig
import io.github.lene.qa.Perf
import io.github.lene.qa.PerfGate
import io.github.lene.qa.RelativeBenchmark
import io.github.lene.qa.Side
import menger.common.Color
import menger.common.ImageSize
import menger.common.ProfilingConfig
import menger.objects.higher_d.Face4D
import menger.objects.higher_d.Mesh4D
import menger.objects.higher_d.Mesh4DGpuFlatten
import menger.objects.higher_d.Mesh4DProjection
import menger.objects.higher_d.TesseractMesh
import menger.objects.higher_d.TesseractSpongeMesh
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.Tag
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object Project4DGpuSuiteTags:
  object Slow extends Tag("Slow")

/** Sprint 18.3 Cut D — equivalence + generality + perf-smoke for the GPU
  * 4D rotation/projection kernel against the existing CPU
  * `Mesh4DProjection.toTriangleMesh` path.
  */
class Project4DGpuSuite extends AnyFlatSpec
    with Matchers
    with LazyLogging
    with PerfGate
    with BeforeAndAfterEach:

  import Project4DGpuSuiteTags.Slow

  given ProfilingConfig = ProfilingConfig.disabled

  private val ImgSize = ImageSize(256, 192)
  private val MaxAbsPixelDiff = 6  // L∞ over RGB; conservative for float32 path divergence.

  // Perf gates (tag Perf): time ratio subject / reference, judged by RelativeBenchmark. Both
  // assert the tests' claim "faster than" (ratio < 1). Measured on the RTX A1000 laptop (idle /
  // under a 99% GPU burn plus CPU load, 2026-09-23), highest upper confidence bound: GPU flatten
  // 0.41; update vs rebuild 0.06 idle but 0.89 loaded -- update is GPU-bound, rebuild mostly
  // CPU, so GPU contention moves only one side. The perf suite's GPU preflight skips the run
  // when another compute process holds the GPU.
  private val MaxRatioGpuFlattenVsCpu = 1.0
  private val MaxRatioUpdateVsRebuild = 1.0
  private val AnimationFrames = 40
  private val AnimationStepDegrees = 9f

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[OptiXRenderer] = None

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def renderer: OptiXRenderer = rendererOpt.getOrElse(
    throw new IllegalStateException("Renderer not initialized")
  )

  override def withFixture(test: NoArgTest): Outcome =
    if rendererOpt.isDefined then super.withFixture(test)
    else cancel("OptiX native library not available")

  override def beforeEach(): Unit =
    super.beforeEach()
    try
      // Touch the companion first so liboptixjni is actually loaded before any
      // @native call — otherwise the suite depends on test ordering to have
      // initialized it (running this suite standalone used to cancel).
      val _ = OptiXRenderer.isLibraryLoaded
      val r = new OptiXRenderer()
      r.initialize()
      rendererOpt = Some(r)
      setupDefaults(r)
    catch case _: Throwable => ()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally
      rendererOpt = None
      super.afterEach()

  private def setupDefaults(r: OptiXRenderer): Unit =
    r.setCamera(
      Vector[3](0.0f, 0.0f, 4.0f),
      Vector[3](0.0f, 0.0f, 0.0f),
      Vector[3](0.0f, 1.0f, 0.0f),
      45.0f
    )
    r.setLight(Vector[3](0.4f, 0.6f, -0.5f), 1.0f)
    r.setSphere(Vector[3](100f, 100f, 100f), 0.01f)  // off-screen
    r.clearPlanes()

  private def opaqueGrey: Material = Material(Color(0.7f, 0.7f, 0.7f, 1.0f), ior = 1.0f)

  private def renderCpu(projection: Mesh4DProjection): Array[Byte] =
    val mesh = projection.toTriangleMesh
    renderer.setTriangleMesh(mesh)
    renderer.addTriangleMeshInstance(Vector[3](0f, 0f, 0f), opaqueGrey, -1)
    val img = renderer.render(ImgSize)
    if img == null then fail(s"CPU render returned null for ${ImgSize.width}x${ImgSize.height}") // scalafix:ok DisableSyntax.null
    img

  private def renderGpu(
    mesh4D: Mesh4D, eyeW: Float, screenW: Float,
    rotXW: Float, rotYW: Float, rotZW: Float
  ): Array[Byte] =
    val quads = Mesh4DGpuFlatten.quadsBuffer(mesh4D)
    renderer.setTriangleMesh4DQuads(
      quads, uvs = null, eyeW = eyeW, screenW = screenW, // scalafix:ok DisableSyntax.null
      rotXW = rotXW, rotYW = rotYW, rotZW = rotZW,
      centerX = 0f, centerY = 0f, centerZ = 0f
    )
    renderer.addTriangleMeshInstance(Vector[3](0f, 0f, 0f), opaqueGrey, -1)
    val img = renderer.render(ImgSize)
    if img == null then fail(s"GPU render returned null for ${ImgSize.width}x${ImgSize.height}") // scalafix:ok DisableSyntax.null
    img

  private def maxAbsRgbDiff(a: Array[Byte], b: Array[Byte]): Int =
    require(a.length == b.length, s"image size mismatch: ${a.length} vs ${b.length}")
    val n = a.length
    val it = (0 until n).iterator.filter(i => i % 4 != 3)  // skip alpha channel
    it.map(i => math.abs((a(i) & 0xFF) - (b(i) & 0xFF))).maxOption.getOrElse(0)

  private def hasVariation(pixels: Array[Byte]): Boolean =
    val r0 = pixels(0) & 0xFF
    val g0 = pixels(1) & 0xFF
    val b0 = pixels(2) & 0xFF
    pixels.grouped(4).exists { px =>
      val dr = math.abs((px(0) & 0xFF) - r0)
      val dg = math.abs((px(1) & 0xFF) - g0)
      val db = math.abs((px(2) & 0xFF) - b0)
      dr + dg + db > 8
    }

  // --- Test 1: equivalence on the existing tesseract CPU pipeline -----------

  "GPU 4D projection" should "match CPU pixels for a non-rotated tesseract" taggedAs Slow in:
    val cpuProj = TesseractMesh(
      center = Vector[3](0f, 0f, 0f), size = 1.0f,
      rotXW = 0f, rotYW = 0f, rotZW = 0f
    )
    val cpuPixels = renderCpu(cpuProj)
    afterEach()
    beforeEach()
    val gpuPixels = renderGpu(
      cpuProj.mesh4D,
      eyeW = cpuProj.eyeW, screenW = cpuProj.screenW,
      rotXW = 0f, rotYW = 0f, rotZW = 0f
    )
    val diff = maxAbsRgbDiff(cpuPixels, gpuPixels)
    logger.info(f"tesseract no-rot L∞ diff: $diff")
    diff should be <= MaxAbsPixelDiff

  it should "match CPU pixels for a rotated tesseract" taggedAs Slow in:
    val cpuProj = TesseractMesh(
      center = Vector[3](0f, 0f, 0f), size = 1.0f,
      rotXW = 12f, rotYW = 18f, rotZW = 7f
    )
    val cpuPixels = renderCpu(cpuProj)
    afterEach()
    beforeEach()
    val gpuPixels = renderGpu(
      cpuProj.mesh4D,
      eyeW = cpuProj.eyeW, screenW = cpuProj.screenW,
      rotXW = 12f, rotYW = 18f, rotZW = 7f
    )
    val diff = maxAbsRgbDiff(cpuPixels, gpuPixels)
    logger.info(f"tesseract rotated L∞ diff: $diff")
    diff should be <= MaxAbsPixelDiff

  // --- Test 2: generality — handcrafted non-tesseract Mesh4D ---------------

  it should "render a handcrafted non-tesseract 4D mesh to non-uniform pixels" taggedAs Slow in:
    val mesh4D = new Mesh4D:
      type V = 4
      override def vertices: Seq[menger.common.Vector[4]] = faces.flatMap(_.asSeq).distinct
      override lazy val faces: Seq[Face4D[4]] = Seq(
        // Quad in the XY plane (z=0, w=0)
        Face4D(
          Vector[4](-0.4f, -0.4f, 0f, 0f), Vector[4](0.4f, -0.4f, 0f, 0f),
          Vector[4](0.4f,  0.4f, 0f, 0f), Vector[4](-0.4f, 0.4f, 0f, 0f)
        ),
        // Quad in the XW plane (y=0, z=0), offset along Z so it stays visible
        Face4D(
          Vector[4](-0.3f, 0f, 0.5f, -0.3f), Vector[4](0.3f, 0f, 0.5f, -0.3f),
          Vector[4](0.3f,  0f, 0.5f,  0.3f), Vector[4](-0.3f, 0f, 0.5f,  0.3f)
        )
      )

    val pixels = renderGpu(mesh4D, eyeW = 3.0f, screenW = 1.5f,
      rotXW = 8f, rotYW = 0f, rotZW = 0f)
    pixels.length shouldBe (ImgSize.width * ImgSize.height * 4)
    hasVariation(pixels) shouldBe true
    renderer.getInstanceCount() shouldBe 1

  // --- Test 3: perf smoke on tesseract-sponge level=2 ----------------------

  // Level 1, not 2: the CPU path takes seconds per op at level 2, times ~34 timed samples.
  it should "set up a tesseract sponge faster via GPU flatten than via CPU" taggedAs Perf in:
    def sponge() = TesseractSpongeMesh(
      center = Vector[3](0f, 0f, 0f), size = 1.0f, level = 1f,
      rotXW = 12f, rotYW = 18f, rotZW = 7f
    )
    // The GPU flatten step skips per-vertex matrix multiplications and normal
    // cross-products, so it should comfortably beat the CPU `toTriangleMesh`
    // path even before the kernel launch is amortised.
    val cpu = Side("CPU toTriangleMesh", () => { val _ = sponge().toTriangleMesh })
    val gpu = Side("GPU flatten", () => { val _ = Mesh4DGpuFlatten.quadsBuffer(sponge().mesh4D) })
    val verdict = RelativeBenchmark.compare(cpu, gpu, MaxRatioGpuFlattenVsCpu, BenchConfig.JvmCpu)
    assertWithin("tesseract sponge L1 setup, GPU flatten vs CPU", verdict)

  // --- Test 4: update equivalence — frame B via update vs from-scratch ------

  it should "match from-scratch render after updateMesh4DProjection rotates" taggedAs Slow in:
    val rotA = (0f, 0f, 0f)
    val rotB = (12f, 18f, 7f)
    val tess = TesseractMesh(
      center = Vector[3](0f, 0f, 0f), size = 1.0f,
      rotXW = rotB._1, rotYW = rotB._2, rotZW = rotB._3
    )
    val quads = Mesh4DGpuFlatten.quadsBuffer(tess.mesh4D)
    // frame A then update to B
    val meshIdx = renderer.setTriangleMesh4DQuads(
      quads, uvs = null, eyeW = tess.eyeW, screenW = tess.screenW, // scalafix:ok DisableSyntax.null
      rotXW = rotA._1, rotYW = rotA._2, rotZW = rotA._3,
      centerX = 0f, centerY = 0f, centerZ = 0f
    )
    renderer.addTriangleMeshInstance(Vector[3](0f, 0f, 0f), opaqueGrey, -1)
    val _ = renderer.render(ImgSize)
    renderer.updateMesh4DProjection(
      meshIdx, eyeW = tess.eyeW, screenW = tess.screenW,
      rotXW = rotB._1, rotYW = rotB._2, rotZW = rotB._3
    )
    val updatedPixels = renderer.render(ImgSize)
    afterEach()
    beforeEach()
    val freshPixels = renderGpu(
      tess.mesh4D, eyeW = tess.eyeW, screenW = tess.screenW,
      rotXW = rotB._1, rotYW = rotB._2, rotZW = rotB._3
    )
    val diff = maxAbsRgbDiff(updatedPixels, freshPixels)
    logger.info(f"update-equivalence L∞ diff: $diff")
    diff should be <= MaxAbsPixelDiff

  // --- Test 5: update perf — animation update vs rebuild --------------------

  it should "animate 4D rotation faster via projection update than rebuild" taggedAs Perf in:
    def sponge(angle: Float) = TesseractSpongeMesh(
      center = Vector[3](0f, 0f, 0f), size = 1.0f, level = 1f,
      rotXW = angle, rotYW = 0f, rotZW = 0f
    )
    def upload(proj: Mesh4DProjection, angle: Float): Int =
      renderer.setTriangleMesh4DQuads(
        Mesh4DGpuFlatten.quadsBuffer(proj.mesh4D), uvs = null, // scalafix:ok DisableSyntax.null
        eyeW = proj.eyeW, screenW = proj.screenW,
        rotXW = angle, rotYW = 0f, rotZW = 0f,
        centerX = 0f, centerY = 0f, centerZ = 0f
      )
    val base = sponge(0f)
    val meshIdx = AtomicInteger(-1)
    val frame = AtomicInteger(0)
    def nextAngle(): Float = (frame.incrementAndGet() % AnimationFrames + 1) * AnimationStepDegrees
    // Uploads append meshes: every sample starts from a scene holding just the base mesh.
    def resetScene(): Unit =
      renderer.clearAllInstances()
      renderer.clearTriangleMesh()
      meshIdx.set(upload(base, 0f))
      renderer.addTriangleMeshInstance(Vector[3](0f, 0f, 0f), opaqueGrey, -1)
    val update = Side(
      "updateMesh4DProjection",
      () => renderer.updateMesh4DProjection(
        meshIdx.get, eyeW = base.eyeW, screenW = base.screenW,
        rotXW = nextAngle(), rotYW = 0f, rotZW = 0f
      ),
      prepare = () => resetScene()
    )
    val rebuild = Side(
      "rebuild",
      () => { val angle = nextAngle(); val _ = upload(sponge(angle), angle) },
      prepare = () => resetScene()
    )
    val verdict = RelativeBenchmark.compare(rebuild, update, MaxRatioUpdateVsRebuild)
    assertWithin("tesseract sponge L1 animation, update vs rebuild", verdict)

  // --- Test 6: return-code contract — setTriangleMesh4DQuads ---

  it should "return a non-negative mesh index for a valid upload" in:
    val proj = TesseractMesh(
      center = Vector[3](0f, 0f, 0f), size = 1.0f,
      rotXW = 0f, rotYW = 0f, rotZW = 0f
    )
    val quads = Mesh4DGpuFlatten.quadsBuffer(proj.mesh4D)
    val meshIdx = renderer.setTriangleMesh4DQuads(
      quads, uvs = null, // scalafix:ok DisableSyntax.null
      eyeW = proj.eyeW, screenW = proj.screenW,
      rotXW = 0f, rotYW = 0f, rotZW = 0f
    )
    meshIdx should be >= 0

  // --- Test 7: sanitizer scene coverage — GAS lifecycle for non-sphere kinds -----
  // Sprint 36 E1: setupDefaults only ever adds a sphere, so gas_registry never held a
  // cylinder/cone/plane/curve alias and the compute-sanitizer gate (memcheck.sh) had
  // nothing to double-free when dispose() -> clearAllInstances() ran on teardown. This
  // test's only job is scene coverage, not pixel output. Under plain `sbt test` it
  // passes cleanly (the double-free is silently absorbed and only logged natively —
  // exactly QA_INCIDENTS.md entry #1's "why didn't it fail" symptom). Under
  // compute-sanitizer instrumentation the wrapped JVM exits non-zero on teardown —
  // expected, and already handled: memcheck.sh treats a non-zero test-run exit as a
  // real failure only when the sanitizer log has no findings; here it does, so the
  // suite reports a non-blocking WARNING (findings attributed to liboptixjni) instead.

  it should "add and clear cylinder/cone/plane/curve instances without native error" in:
    renderer.addCylinderInstance(
      Vector[3](-2.0f, -1.0f, 0.0f), Vector[3](-2.0f, 1.0f, 0.0f), 0.3f, opaqueGrey)
    renderer.addConeInstance(
      Vector[3](2.0f, 1.0f, 0.0f), Vector[3](2.0f, -1.0f, 0.0f), 0.4f, opaqueGrey)
    renderer.addPlaneInstance(Vector[3](0.0f, 1.0f, 0.0f), -2.0f, opaqueGrey)
    val points = Array(0.0f, -1.0f, 1.0f, 0.3f, 0.0f, 1.0f, 0.6f, 1.0f, 1.0f, 0.9f, 2.0f, 1.0f)
    renderer.addCurveInstance(points, Array.fill(4)(0.1f), opaqueGrey)
    val _ = renderer.render(ImgSize)
    renderer.getInstanceCount() shouldBe 4
