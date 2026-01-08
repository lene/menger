package menger

import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.common.Color
import menger.common.ImageSize
import menger.objects.SpongeBySurface
import menger.objects.SpongeByVolume
import menger.optix.OptiXRenderer
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Tag
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object Slow extends Tag("Slow")

class SpongePerformanceSuite extends AnyFlatSpec
    with Matchers
    with LazyLogging
    with BeforeAndAfterEach:

  private val runningUnderSanitizer: Boolean =
    sys.env.get("RUNNING_UNDER_COMPUTE_SANITIZER").contains("true")

  given ProfilingConfig = ProfilingConfig.disabled

  private val STANDARD_IMAGE_SIZE = ImageSize(800, 600)

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[OptiXRenderer] = None

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  protected def renderer: OptiXRenderer = rendererOpt.getOrElse(
    throw new IllegalStateException("Renderer not initialized")
  )

  override def beforeEach(): Unit =
    super.beforeEach()
    require(OptiXRenderer.isLibraryLoaded, "OptiX native library failed to load")
    val r = new OptiXRenderer()
    r.initialize()
    rendererOpt = Some(r)
    setupDefaults()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally super.afterEach()
    rendererOpt = None

  protected def setupDefaults(): Unit =
    import menger.common.Vector
    renderer.setCamera(
      Vector[3](0.0f, 0.5f, 3.0f),
      Vector[3](0.0f, 0.0f, 0.0f),
      Vector[3](0.0f, 1.0f, 0.0f),
      60.0f
    )
    renderer.setLight(Vector[3](0.5f, 0.5f, -0.5f), 1.0f)
    renderer.setSphere(Vector[3](0.0f, 0.0f, 0.0f), 0.5f)

  private def measureTimeMs[T](block: => T): (T, Double) =
    val startNs = System.nanoTime()
    val result = block
    val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0
    (result, elapsedMs)

  "Sponge mesh generation" should "generate level 0 surface sponge quickly" taggedAs Slow in:
    assume(!runningUnderSanitizer, "Performance test skipped under compute-sanitizer")

    val (mesh, genTimeMs) = measureTimeMs:
      SpongeBySurface(
        center = Vector3.Zero,
        scale = 2.0f,
        level = 0f
      ).toTriangleMesh

    logger.info(f"Level 0 surface sponge: ${mesh.numTriangles} triangles in $genTimeMs%.2fms")
    mesh.numTriangles shouldBe 12 // 6 faces * 2 triangles/face
    genTimeMs should be < 100.0

  it should "generate level 1 surface sponge quickly" taggedAs Slow in:
    assume(!runningUnderSanitizer, "Performance test skipped under compute-sanitizer")

    val (mesh, genTimeMs) = measureTimeMs:
      SpongeBySurface(
        center = Vector3.Zero,
        scale = 2.0f,
        level = 1f
      ).toTriangleMesh

    logger.info(f"Level 1 surface sponge: ${mesh.numTriangles} triangles in $genTimeMs%.2fms")
    mesh.numTriangles shouldBe 144 // 12 faces/face * 6 faces * 2 triangles
    genTimeMs should be < 200.0

  it should "generate level 2 surface sponge within 1 second" taggedAs Slow in:
    assume(!runningUnderSanitizer, "Performance test skipped under compute-sanitizer")

    val (mesh, genTimeMs) = measureTimeMs:
      SpongeBySurface(
        center = Vector3.Zero,
        scale = 2.0f,
        level = 2f
      ).toTriangleMesh

    logger.info(f"Level 2 surface sponge: ${mesh.numTriangles} triangles in $genTimeMs%.2fms")
    mesh.numTriangles shouldBe 1728 // 144 sub-faces/face * 6 faces * 2 triangles
    genTimeMs should be < 1000.0

  it should "generate level 3 surface sponge within 5 seconds" taggedAs Slow in:
    assume(!runningUnderSanitizer, "Performance test skipped under compute-sanitizer")

    val (mesh, genTimeMs) = measureTimeMs:
      SpongeBySurface(
        center = Vector3.Zero,
        scale = 2.0f,
        level = 3f
      ).toTriangleMesh

    logger.info(f"Level 3 surface sponge: ${mesh.numTriangles} triangles in $genTimeMs%.2fms")
    genTimeMs should be < 5000.0

  "Sponge volume mesh generation" should "generate level 0 volume sponge quickly" taggedAs Slow in:
    assume(!runningUnderSanitizer, "Performance test skipped under compute-sanitizer")

    val (mesh, genTimeMs) = measureTimeMs:
      SpongeByVolume(
        center = Vector3.Zero,
        scale = 2.0f,
        level = 0f
      ).toTriangleMesh

    logger.info(f"Level 0 volume sponge: ${mesh.numTriangles} triangles in $genTimeMs%.2fms")
    mesh.numTriangles shouldBe 12
    genTimeMs should be < 100.0

  it should "generate level 1 volume sponge quickly" taggedAs Slow in:
    assume(!runningUnderSanitizer, "Performance test skipped under compute-sanitizer")

    val (mesh, genTimeMs) = measureTimeMs:
      SpongeByVolume(
        center = Vector3.Zero,
        scale = 2.0f,
        level = 1f
      ).toTriangleMesh

    logger.info(f"Level 1 volume sponge: ${mesh.numTriangles} triangles in $genTimeMs%.2fms")
    mesh.numTriangles shouldBe 240
    genTimeMs should be < 200.0

  it should "generate level 2 volume sponge within 1 second" taggedAs Slow in:
    assume(!runningUnderSanitizer, "Performance test skipped under compute-sanitizer")

    val (mesh, genTimeMs) = measureTimeMs:
      SpongeByVolume(
        center = Vector3.Zero,
        scale = 2.0f,
        level = 2f
      ).toTriangleMesh

    logger.info(f"Level 2 volume sponge: ${mesh.numTriangles} triangles in $genTimeMs%.2fms")
    mesh.numTriangles shouldBe 4800
    genTimeMs should be < 1000.0

  "Sponge rendering performance" should "render level 2 surface sponge at >1 FPS (800x600)" taggedAs Slow in:
    assume(!runningUnderSanitizer, "Performance test skipped under compute-sanitizer")

    val mesh = SpongeBySurface(
      center = Vector3.Zero,
      scale = 2.0f,
      level = 2f
    ).toTriangleMesh

    renderer.setTriangleMesh(mesh)
    renderer.setTriangleMeshColor(Color(0.8f, 0.8f, 0.8f))
    renderer.setTriangleMeshIOR(1.0f)
    renderer.setPlane(1, true, -2.0f)

    val renderSize = STANDARD_IMAGE_SIZE
    val iterations = 10

    // Warmup
    renderer.render(renderSize)

    val (_, totalMs) = measureTimeMs:
      (0 until iterations).foreach(_ => renderer.render(renderSize))

    val avgMs = totalMs / iterations
    val fps = 1000.0 / avgMs

    logger.info(f"Level 2 surface sponge render: ${mesh.numTriangles} triangles, ${renderSize.width}x${renderSize.height}, $avgMs%.2fms/frame ($fps%.1f FPS)")
    fps should be > 1.0

  it should "render level 2 volume sponge at >1 FPS (800x600)" taggedAs Slow in:
    assume(!runningUnderSanitizer, "Performance test skipped under compute-sanitizer")

    val mesh = SpongeByVolume(
      center = Vector3.Zero,
      scale = 2.0f,
      level = 2f
    ).toTriangleMesh

    renderer.setTriangleMesh(mesh)
    renderer.setTriangleMeshColor(Color(0.8f, 0.8f, 0.8f))
    renderer.setTriangleMeshIOR(1.0f)
    renderer.setPlane(1, true, -2.0f)

    val renderSize = STANDARD_IMAGE_SIZE
    val iterations = 10

    // Warmup
    renderer.render(renderSize)

    val (_, totalMs) = measureTimeMs:
      (0 until iterations).foreach(_ => renderer.render(renderSize))

    val avgMs = totalMs / iterations
    val fps = 1000.0 / avgMs

    logger.info(f"Level 2 volume sponge render: ${mesh.numTriangles} triangles, ${renderSize.width}x${renderSize.height}, $avgMs%.2fms/frame ($fps%.1f FPS)")
    fps should be > 1.0

  it should "render transparent level 1 surface sponge at >1 FPS (800x600)" taggedAs Slow in:
    assume(!runningUnderSanitizer, "Performance test skipped under compute-sanitizer")

    val mesh = SpongeBySurface(
      center = Vector3.Zero,
      scale = 2.0f,
      level = 1f
    ).toTriangleMesh

    renderer.setTriangleMesh(mesh)
    renderer.setTriangleMeshColor(Color(0.9f, 0.9f, 1.0f, 0.5f))
    renderer.setTriangleMeshIOR(1.5f)
    renderer.setPlane(1, true, -2.0f)

    val renderSize = STANDARD_IMAGE_SIZE
    val iterations = 10

    // Warmup
    renderer.render(renderSize)

    val (_, totalMs) = measureTimeMs:
      (0 until iterations).foreach(_ => renderer.render(renderSize))

    val avgMs = totalMs / iterations
    val fps = 1000.0 / avgMs

    logger.info(f"Level 1 transparent surface sponge render: ${mesh.numTriangles} triangles, ${renderSize.width}x${renderSize.height}, $avgMs%.2fms/frame ($fps%.1f FPS)")
    fps should be > 1.0
