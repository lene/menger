package examples.dsl.ladder.variants

import examples.dsl.ladder.Choreography
import examples.dsl.ladder.Choreography.Kind
import examples.dsl.ladder.DimensionLadder
import examples.dsl.ladder.Look
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MomentSuite extends AnyFlatSpec with Matchers:

  private val tops = 4 to 8
  private val fractions = List(0f, 0.1f, 0.25f, 0.5f, 0.9f, 1f)
  private def timeline(top: Int): Choreography =
    Choreography(DimensionLadder.cycleSeconds, top, Look.defaultDepths)

  "Moment.of" should "find the segment of that kind and dimension, the same way into it" in:
    for top <- tops; segment <- timeline(top).segments; u <- fractions do
      withClue(s"top $top, $segment, u $u: "):
        val t = Moment.of(timeline(top), segment.kind, segment.dimension, u)
        val (found, foundU) = timeline(top).segmentAt(t)
        found shouldBe segment
        foundU shouldBe u +- 1e-3f

  it should "return start + u * seconds inside a segment" in:
    val breath = timeline(6).segments.find(s => s.kind == Kind.Breath && s.dimension == 5).get
    Moment.of(timeline(6), Kind.Breath, 5, 0.5f) shouldBe
      breath.start + 0.5f * breath.seconds +- 1e-4f
    Moment.of(timeline(6), Kind.Breath, 5, 0f) shouldBe breath.start

  it should "stay inside the segment at u = 1, where the next one begins" in:
    val fold = timeline(5).segments.find(s => s.kind == Kind.Fold && s.dimension == 4).get
    val t = Moment.of(timeline(5), Kind.Fold, 4, 1f)
    t should be < fold.end
    t shouldBe fold.end +- 1e-3f

  it should "fail with a clear message when the timeline has no such segment" in:
    val e = the[IllegalArgumentException] thrownBy Moment.of(timeline(5), Kind.Breath, 7, 0.5f)
    e.getMessage should include("Breath")
    e.getMessage should include("7")
    an[IllegalArgumentException] should be thrownBy Moment.of(timeline(6), Kind.Unsolidify, 6, 0.5f)
    an[IllegalArgumentException] should be thrownBy Moment.of(timeline(6), Kind.Hold, 4, 0.5f)

  it should "reject an u outside 0 to 1" in:
    List(-0.1f, 1.1f, Float.NaN).foreach: u =>
      an[IllegalArgumentException] should be thrownBy Moment.of(timeline(6), Kind.Fold, 4, u)

  "Moment.ofColumn" should "map a sheet's column index to the moment's time" in:
    val moments = Vector(Moment(Kind.Hold, 3, 0f), Moment(Kind.Breath, 3, 0.5f),
      Moment(Kind.Fold, 4, 0.5f))
    val column = Moment.ofColumn(timeline(6), moments)
    column(0f) shouldBe Moment.of(timeline(6), Kind.Hold, 3, 0f)
    column(1f) shouldBe Moment.of(timeline(6), Kind.Breath, 3, 0.5f)
    column(2f) shouldBe Moment.of(timeline(6), Kind.Fold, 4, 0.5f)

  it should "reject a column the sheet has no moment for" in:
    val column = Moment.ofColumn(timeline(6), Vector(Moment(Kind.Hold, 3, 0f)))
    List(-1f, 1f, 0.5f).foreach(c => an[IllegalArgumentException] should be thrownBy column(c))

  "The example variant" should "show its moments in the column order, and nothing else" in:
    val kinds = List(0f, 1f, 2f).map(t => Example.moments(t.toInt).kind)
    kinds shouldBe List(Kind.Hold, Kind.Breath, Kind.Fold)
    val scene = Example.scene(1f)
    scene.objects.last.toObjectSpec.objectType shouldBe "sponge-volume"
    Example.scene(0f).objects.last.toObjectSpec.objectType shouldBe "cube"
