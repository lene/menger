package menger.engines.scene

import menger.geometry.LevelMix
import menger.geometry.MengerRenderer
import io.github.lene.optix.OptiXRenderer
import menger.cli.ObjectSpecParser
import menger.spec.ObjectSpec
import menger.common.ImageSize
import menger.common.ProfilingConfig
import menger.common.Vector
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.tagobjects.Slow

/** A refractive fractional-level object is drawn as its two whole levels, rendered separately
  * and mixed: the frame must equal the weighted mix of renders of the pure levels, for one
  * object and for two whose light paths cross, where every combination of their levels counts
  * (usability review 2026-10, session 4, F85; lene/menger#69). */
class LevelMixRenderSuite extends AnyFlatSpec with Matchers with BeforeAndAfterEach:
  given ProfilingConfig = ProfilingConfig.disabled

  private val Size = ImageSize(200, 150)
  private val MaxInstances = 64
  // Mean per-channel difference, 0..255: the mix rounds once per pixel.
  private val SameImage = 0.5

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[MengerRenderer] = None

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def renderer: MengerRenderer =
    rendererOpt.getOrElse(throw new IllegalStateException("Renderer not initialized"))

  override def withFixture(test: NoArgTest): Outcome =
    if rendererOpt.isDefined then super.withFixture(test)
    else cancel("OptiX native library not available")

  override def beforeEach(): Unit =
    super.beforeEach()
    try
      val _ = OptiXRenderer.isLibraryLoaded
      val r = MengerRenderer()
      if r.initialize(MaxInstances) then
        r.setCamera(Vector[3](0.4f, 0.6f, 4f), Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f), 45f)
        r.setLight(Vector[3](0.4f, -0.6f, -0.5f), 1f)
        rendererOpt = Some(r)
    catch case _: Throwable => ()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally
      rendererOpt = None
      super.afterEach()

  private def spec(definition: String): ObjectSpec =
    ObjectSpecParser.parse(definition).fold(e => fail(e), identity)

  private def render(specs: List[ObjectSpec], builder: SceneBuilder = TriangleMeshSceneBuilder(".")) =
    renderer.clearAllInstances()
    builder.validateAndBuild(specs, renderer, MaxInstances).get
    renderer.render(Size)

  private def meanDifference(a: Array[Byte], b: Array[Byte]): Double =
    a.indices.map(i => math.abs((a(i) & 0xff) - (b(i) & 0xff))).sum.toDouble / a.length

  private def glassSponge(level: Float) =
    spec(s"type=sponge-volume:level=$level:size=1.5:material=glass")

  private def channel(image: Array[Byte], pixel: Int, c: Int): Int = image(pixel * 4 + c) & 0xff

  // Pixels where two objects' levels interact: |i11 - i10 - i01 + i00| above `threshold` in
  // some colour channel, i.e. where no sum of one-object changes explains the image.
  private def crossTermPixels(
    i00: Array[Byte], i10: Array[Byte], i01: Array[Byte], i11: Array[Byte], threshold: Int
  ): IndexedSeq[Int] =
    (0 until i00.length / 4).filter { p =>
      (0 until 3).exists { c =>
        math.abs(channel(i11, p, c) - channel(i10, p, c) - channel(i01, p, c) + channel(i00, p, c)) >
          threshold
      }
    }

  private def regionDifference(a: Array[Byte], b: Array[Byte], pixels: Seq[Int]): Double =
    pixels.map(p => (0 until 3).map(c => math.abs(channel(a, p, c) - channel(b, p, c))).sum)
      .sum.toDouble / (pixels.size * 3)

  "A refractive fractional sponge" should "render as the linear mix of its two whole levels" taggedAs Slow in:
    val mixed = render(List(glassSponge(1.25f)))
    renderer.levelMixCount shouldBe 1
    val expected = LevelMix.mix(Seq(
      (render(List(glassSponge(1f))), 0.75),
      (render(List(glassSponge(2f))), 0.25)
    ))
    meanDifference(mixed, expected) should be < SameImage

  // The 2^k case. The two sponges sit on the view axis, so the back one's centre hole shows
  // through the front one's only when both are at their upper level: a cross term, which a
  // first-order mix (k + 1 renders, each object changed alone) misses. Compared where the cross
  // term is large -- over the whole frame it is diluted below the rounding tolerance.
  "Two refractive fractional sponges whose light paths cross" should
    "render as the mix of all four combinations of their levels" taggedAs Slow in:
      renderer.setCamera(Vector[3](0f, 0f, 5f), Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f), 45f)
      def back(level: Float) = spec(s"type=sponge-volume:level=$level:size=1.8:material=glass")
      def front(level: Float) =
        spec(s"type=sponge-surface:level=$level:pos=0,0,1.5:size=0.9:material=glass")
      val (fBack, fFront) = (0.25, 0.75)
      val mixed = render(List(back(0.25f), front(0.75f)))
      renderer.levelMixCount shouldBe 2
      def pure(backLevel: Int, frontLevel: Int) =
        render(List(back(backLevel.toFloat), front(frontLevel.toFloat)))
      val (i00, i10, i01, i11) = (pure(0, 0), pure(1, 0), pure(0, 1), pure(1, 1))
      val exact = LevelMix.mix(Seq(
        (i00, (1 - fBack) * (1 - fFront)), (i10, fBack * (1 - fFront)),
        (i01, (1 - fBack) * fFront), (i11, fBack * fFront)
      ))
      val firstOrder = LevelMix.mix(Seq((i00, 1 - fBack - fFront), (i10, fBack), (i01, fFront)))
      val crossing = crossTermPixels(i00, i10, i01, i11, threshold = 8)
      info(s"${crossing.size} of ${Size.width * Size.height} pixels show the cross term")
      crossing.size should be > (Size.width * Size.height / 50)
      // A first-order mixer must miss the tolerance by a wide margin, or this proves nothing.
      regionDifference(firstOrder, exact, crossing) should be > (4 * SameImage)
      regionDifference(mixed, exact, crossing) should be < SameImage

  "A refractive fractional 4D IFS object" should "render as the mix of its two whole levels" taggedAs Slow in:
    def sierpinski(level: Float) = spec(s"type=sierpinski4d:level=$level:size=1.4:material=glass")
    val builder = Instanced4DSceneBuilder(IFS4DType.Sierpinski4D)
    val mixed = render(List(sierpinski(2.5f)), builder)
    renderer.levelMixCount shouldBe 1
    val expected = LevelMix.mix(Seq(
      (render(List(sierpinski(2f)), builder), 0.5),
      (render(List(sierpinski(3f)), builder), 0.5)
    ))
    meanDifference(mixed, expected) should be < SameImage

  // Opaque caps hide what lies behind them: fading them already mixes the levels in one render.
  "An opaque fractional sponge" should "keep its fading hole caps, with no level mix" in:
    val _ = render(List(spec("type=sponge-volume:level=1.25:size=1.5:material=matte")))
    renderer.levelMixCount shouldBe 0

  "clearAllInstances" should "forget the level mixes" in:
    val _ = render(List(glassSponge(1.25f)))
    renderer.levelMixCount shouldBe 1
    renderer.clearAllInstances()
    renderer.levelMixCount shouldBe 0

  // The animated window changes a level within [n, n+1) in place: the mix fraction follows.
  "An in-place level change of a two-level 4D sponge" should "render like a fresh build" taggedAs Slow in:
    def film(level: Float) = spec(s"type=tesseract-sponge:level=$level:size=1.5:material=film")
    renderer.clearAllInstances()
    val state = TrackedMesh4D.build(List(film(1.2f)), renderer, ".", _ => MaxInstances).get
      .getOrElse(fail("the scene was not tracked"))
    renderer.levelMixCount shouldBe 1
    val _ = TrackedMesh4D.updateInPlace(state, List(film(1.8f)), renderer)
    val moved = renderer.render(Size)
    renderer.clearAllInstances()
    val _ = TrackedMesh4D.build(List(film(1.8f)), renderer, ".", _ => MaxInstances).get
    meanDifference(moved, renderer.render(Size)) should be < SameImage
