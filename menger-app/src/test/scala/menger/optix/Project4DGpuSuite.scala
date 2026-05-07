package menger.optix

import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.ProfilingConfig
import menger.common.Color
import menger.common.ImageSize
import menger.common.Vector
import menger.objects.higher_d.Face4D
import menger.objects.higher_d.Mesh4D
import menger.objects.higher_d.Mesh4DGpuFlatten
import menger.objects.higher_d.Mesh4DProjection
import menger.objects.higher_d.TesseractMesh
import menger.objects.higher_d.TesseractSpongeMesh
import org.scalatest.BeforeAndAfterEach
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
    with BeforeAndAfterEach:

  import Project4DGpuSuiteTags.Slow

  given ProfilingConfig = ProfilingConfig.disabled

  private val ImgSize = ImageSize(256, 192)
  private val MaxAbsPixelDiff = 6  // L∞ over RGB; conservative for float32 path divergence.

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[OptiXRenderer] = None

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def renderer: OptiXRenderer = rendererOpt.getOrElse(
    throw new IllegalStateException("Renderer not initialized")
  )

  override def beforeEach(): Unit =
    super.beforeEach()
    require(OptiXRenderer.isLibraryLoaded, "OptiX native library failed to load")
    val r = new OptiXRenderer()
    r.initialize()
    rendererOpt = Some(r)
    setupDefaults(r)

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
    img.getOrElse(fail(s"CPU render returned None for ${ImgSize.width}x${ImgSize.height}"))

  private def renderGpu(
    mesh4D: Mesh4D, eyeW: Float, screenW: Float,
    rotXW: Float, rotYW: Float, rotZW: Float
  ): Array[Byte] =
    val quads = Mesh4DGpuFlatten.quadsBuffer(mesh4D)
    renderer.setTriangleMesh4DQuads(
      quads, uvs = None, eyeW = eyeW, screenW = screenW,
      rotXW = rotXW, rotYW = rotYW, rotZW = rotZW,
      centerX = 0f, centerY = 0f, centerZ = 0f
    )
    renderer.addTriangleMeshInstance(Vector[3](0f, 0f, 0f), opaqueGrey, -1)
    val img = renderer.render(ImgSize)
    img.getOrElse(fail(s"GPU render returned None for ${ImgSize.width}x${ImgSize.height}"))

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
      center = Vector3(0f, 0f, 0f), size = 1.0f,
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
      center = Vector3(0f, 0f, 0f), size = 1.0f,
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

  it should "set up tesseract-sponge level=2 at least as fast on GPU as CPU" taggedAs Slow in:
    val level = 2f

    val (cpuMesh, cpuMs) = measureMs:
      TesseractSpongeMesh(
        center = Vector3(0f, 0f, 0f), size = 1.0f, level = level,
        rotXW = 12f, rotYW = 18f, rotZW = 7f
      ).toTriangleMesh

    val (gpuQuads, gpuFlattenMs) = measureMs:
      val proj = TesseractSpongeMesh(
        center = Vector3(0f, 0f, 0f), size = 1.0f, level = level,
        rotXW = 12f, rotYW = 18f, rotZW = 7f
      )
      Mesh4DGpuFlatten.quadsBuffer(proj.mesh4D)

    logger.info(f"tesseract-sponge L$level%.0f setup — CPU=${cpuMs}%.1fms, GPU-flatten=${gpuFlattenMs}%.1fms; CPU triangles=${cpuMesh.numTriangles}, GPU quads=${gpuQuads.length / 16}")

    // The GPU flatten step skips per-vertex matrix multiplications and normal
    // cross-products, so it should comfortably beat the CPU `toTriangleMesh`
    // path even before the kernel launch is amortised.
    gpuFlattenMs should be <= cpuMs

  // --- Test 4: update equivalence — frame B via update vs from-scratch ------

  it should "match from-scratch render after updateMesh4DProjection rotates" taggedAs Slow in:
    val rotA = (0f, 0f, 0f)
    val rotB = (12f, 18f, 7f)
    val tess = TesseractMesh(
      center = Vector3(0f, 0f, 0f), size = 1.0f,
      rotXW = rotB._1, rotYW = rotB._2, rotZW = rotB._3
    )
    val quads = Mesh4DGpuFlatten.quadsBuffer(tess.mesh4D)
    // frame A then update to B
    val meshIdx = renderer.setTriangleMesh4DQuads(
      quads, uvs = None, eyeW = tess.eyeW, screenW = tess.screenW,
      rotXW = rotA._1, rotYW = rotA._2, rotZW = rotA._3,
      centerX = 0f, centerY = 0f, centerZ = 0f
    )
    renderer.addTriangleMeshInstance(Vector[3](0f, 0f, 0f), opaqueGrey, -1)
    val _ = renderer.render(ImgSize).getOrElse(fail("frame A render returned None"))
    renderer.updateMesh4DProjection(
      meshIdx, eyeW = tess.eyeW, screenW = tess.screenW,
      rotXW = rotB._1, rotYW = rotB._2, rotZW = rotB._3
    )
    val updatedPixels = renderer.render(ImgSize).getOrElse(fail("frame B render returned None"))
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

  it should "animate 4D rotation faster via updateMesh4DProjection than via rebuild" taggedAs Slow in:
    val frames = 10
    val proj0 = TesseractSpongeMesh(
      center = Vector3(0f, 0f, 0f), size = 1.0f, level = 2f,
      rotXW = 0f, rotYW = 0f, rotZW = 0f
    )
    val quads = Mesh4DGpuFlatten.quadsBuffer(proj0.mesh4D)
    val meshIdx = renderer.setTriangleMesh4DQuads(
      quads, uvs = None, eyeW = proj0.eyeW, screenW = proj0.screenW,
      rotXW = 0f, rotYW = 0f, rotZW = 0f,
      centerX = 0f, centerY = 0f, centerZ = 0f
    )
    renderer.addTriangleMeshInstance(Vector[3](0f, 0f, 0f), opaqueGrey, -1)
    val (_, updateMs) = measureMs:
      (0 until frames).foreach { i =>
        val angle = (i + 1) * 9f
        renderer.updateMesh4DProjection(
          meshIdx, eyeW = proj0.eyeW, screenW = proj0.screenW,
          rotXW = angle, rotYW = 0f, rotZW = 0f
        )
      }
    val (_, rebuildMs) = measureMs:
      (0 until frames).foreach { i =>
        val angle = (i + 1) * 9f
        val proj = TesseractSpongeMesh(
          center = Vector3(0f, 0f, 0f), size = 1.0f, level = 2f,
          rotXW = angle, rotYW = 0f, rotZW = 0f
        )
        val q = Mesh4DGpuFlatten.quadsBuffer(proj.mesh4D)
        val _ = renderer.setTriangleMesh4DQuads(
          q, uvs = None, eyeW = proj.eyeW, screenW = proj.screenW,
          rotXW = angle, rotYW = 0f, rotZW = 0f,
          centerX = 0f, centerY = 0f, centerZ = 0f
        )
      }
    logger.info(f"animation $frames frames — update=${updateMs}%.1fms, rebuild=${rebuildMs}%.1fms")
    updateMs should be < rebuildMs

  // --- Test 6: return-code contract — setTriangleMesh4DQuads ---

  it should "return a non-negative mesh index for a valid upload" in:
    val proj = TesseractMesh(
      center = Vector3(0f, 0f, 0f), size = 1.0f,
      rotXW = 0f, rotYW = 0f, rotZW = 0f
    )
    val quads = Mesh4DGpuFlatten.quadsBuffer(proj.mesh4D)
    val meshIdx = renderer.setTriangleMesh4DQuads(
      quads, uvs = None,
      eyeW = proj.eyeW, screenW = proj.screenW,
      rotXW = 0f, rotYW = 0f, rotZW = 0f
    )
    meshIdx should be >= 0

  private def measureMs[T](block: => T): (T, Double) =
    val start = System.nanoTime()
    val result = block
    val ms = (System.nanoTime() - start) / 1_000_000.0
    (result, ms)
