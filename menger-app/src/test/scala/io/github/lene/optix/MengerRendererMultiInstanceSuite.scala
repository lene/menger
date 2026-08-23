package io.github.lene.optix

import com.typesafe.scalalogging.LazyLogging
import menger.common.Color
import menger.common.ImageSize
import menger.common.Material
import menger.common.Vector
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Regression for optix-jni's addCustomGeometryInstance GAS-per-instance fix (Sprint 36
  * #12): a second instance of the same GPU-4D-fast-path type (menger4d / sierpinski4d /
  * hexadecachoron4d) previously never rendered. The bug lived entirely in optix-jni's
  * generic custom-geometry SPI (a shared acceleration structure cached by geometry type,
  * built from whichever instance's AABB arrived first, silently reused for every later
  * instance of that type) and could not be reproduced against optix-jni's own SPI test
  * suite, whose stub shader ignores the AABB/blob for its actual intersection math the way
  * the real menger4d/sierpinski4d/hexadecachoron4d shaders do — so the regression test has
  * to exercise the real shaders, which only exist here, not in optix-jni.
  */
class MengerRendererMultiInstanceSuite extends AnyFlatSpec
    with Matchers
    with LazyLogging
    with BeforeAndAfterEach:

  private val ImgSize = ImageSize(256, 192)
  private val gold = Material(Color(1f, 0.84f, 0f), roughness = 0.2f, metallic = 1f)

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[MengerRenderer] = None

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def renderer: MengerRenderer = rendererOpt.getOrElse(
    throw new IllegalStateException("Renderer not initialized")
  )

  override def withFixture(test: NoArgTest): Outcome =
    if rendererOpt.isDefined then super.withFixture(test)
    else cancel("OptiX native library not available")

  override def beforeEach(): Unit =
    super.beforeEach()
    try
      // Both native libraries must be loaded before any @native call: optix-jni backs
      // OptiXRenderer's own methods (e.g. initializeNative), mengergeometry backs
      // MengerRenderer's 4D-specific additions (Menger4D module loading etc.).
      val _ = OptiXRenderer.isLibraryLoaded
      val _ = MengerRenderer.isLibraryLoaded
      val r = new MengerRenderer()
      r.initialize()
      r.setCamera(
        Vector[3](0f, 3f, 6f), Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f), 45f)
      r.setLight(Vector[3](0.4f, 0.6f, -0.5f), 1.0f)
      r.setSphere(Vector[3](100f, 100f, 100f), 0.01f) // off-screen, keeps the renderer valid
      r.clearPlanes()
      rendererOpt = Some(r)
    catch case _: Throwable => ()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally
      rendererOpt = None
      super.afterEach()

  // Non-background variation in a half of the image proves an instance actually rendered
  // there -- not just that SOME pixel somewhere differs, which a single instance could
  // satisfy on its own even if the other silently never rendered.
  private def hasVariation(pixels: Array[Byte], width: Int, xFrom: Int, xTo: Int): Boolean =
    val r0 = pixels(0) & 0xFF; val g0 = pixels(1) & 0xFF; val b0 = pixels(2) & 0xFF
    (0 until pixels.length / 4).filter(i => (i % width) >= xFrom && (i % width) < xTo).exists { i =>
      val px = i * 4
      val dr = math.abs((pixels(px) & 0xFF) - r0)
      val dg = math.abs((pixels(px + 1) & 0xFF) - g0)
      val db = math.abs((pixels(px + 2) & 0xFF) - b0)
      dr + dg + db > 8
    }

  "MengerRenderer" should "render two menger4d instances at different positions, not just the first" in:
    renderer.addMenger4DInstance(
      2, 2, Vector[3](-1.2f, 0f, 0f), 0.8f, 3.0f, 1.5f, 0f, 0f, 0f, gold)
    renderer.addMenger4DInstance(
      2, 2, Vector[3](1.2f, 0f, 0f), 0.8f, 3.0f, 1.5f, 0f, 0f, 0f, gold)

    val pixels = renderer.render(ImgSize)
    if pixels == null then fail("render returned null") // scalafix:ok DisableSyntax.null
    val leftHalf = hasVariation(pixels, ImgSize.width, 0, ImgSize.width / 2)
    val rightHalf = hasVariation(pixels, ImgSize.width, ImgSize.width / 2, ImgSize.width)
    logger.info(s"two menger4d instances: leftHalf=$leftHalf rightHalf=$rightHalf")
    leftHalf shouldBe true
    rightHalf shouldBe true

  it should "render two sierpinski4d instances at different positions, not just the first" in:
    renderer.addSierpinski4DInstance(
      2, Vector[3](-1.2f, 0f, 0f), 0.8f, 3.0f, 1.5f, 0f, 0f, 0f, gold)
    renderer.addSierpinski4DInstance(
      2, Vector[3](1.2f, 0f, 0f), 0.8f, 3.0f, 1.5f, 0f, 0f, 0f, gold)

    val pixels = renderer.render(ImgSize)
    if pixels == null then fail("render returned null") // scalafix:ok DisableSyntax.null
    val leftHalf = hasVariation(pixels, ImgSize.width, 0, ImgSize.width / 2)
    val rightHalf = hasVariation(pixels, ImgSize.width, ImgSize.width / 2, ImgSize.width)
    logger.info(s"two sierpinski4d instances: leftHalf=$leftHalf rightHalf=$rightHalf")
    leftHalf shouldBe true
    rightHalf shouldBe true
