package examples.dsl.ladder

import examples.dsl.ladder.DimensionLadder.Kind
import menger.dsl.Cylinder
import menger.dsl.Scene
import menger.spec.ObjectSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The object rests at one height for the whole cycle, chosen so that its lowest point over every
  * frame, edge tubes included, just touches the plinth's clearance (menger#160, decided
  * 2026-11-14). The lowest point of a frame is its n-cube's lowest corner, as the sponge inside
  * the cube's hull never reaches lower; both are checked here. */
class RestHeightSuite extends AnyFlatSpec with Matchers:

  private val cycle = DimensionLadder.cycleSeconds
  private val tops = 4 to 8
  private val fps = 30
  private val touching = 1e-4f
  private val noise = 1e-6f
  private val clearance = Stage.plinthClearance

  /** One frame: the scene's object, and how far below its centre its lowest point reaches. */
  private case class Frame(label: String, spec: ObjectSpec, reach: Float)

  private def plinthTop(scene: Scene): Float =
    scene.objects.collectFirst { case c: Cylinder => c }.get.p1.get.y

  // A sponge lies inside its n-cube, and a projection keeps the hull: its level-0 mesh is the hull.
  private def hullOf(spec: ObjectSpec): ObjectSpec = spec.copy(level = spec.level.map(_ => 0f))

  private def lowestY(spec: ObjectSpec): Float = WorldMesh.corners(spec).map(_._2).min

  private def reachOf(spec: ObjectSpec): Float =
    spec.y - lowestY(hullOf(spec)) + spec.edgeRadius.getOrElse(0f)

  // Every frame at 30 fps, and both ends of every segment.
  private def framesOf(top: Int): Seq[Frame] =
    val sampled = (0 until (cycle * fps).toInt).map: i =>
      val t = i.toFloat / fps
      val (segment, u) = DimensionLadder.segmentAt(t, cycle, top)
      (s"t $t, $segment, u $u", DimensionLadder.sceneAt(t, cycle, top).objects.last.toObjectSpec,
        segment, u)
    val ends = DimensionLadder.segments(cycle, top).flatMap: segment =>
      List(0f, 1f).map(u => (s"$segment, u $u",
        DimensionLadder.objectAt(segment, u, cycle, top).toObjectSpec, segment, u))
    (sampled ++ ends).map: (label, spec, segment, u) =>
      // Every stage's eye is in front of every point: this fails if a point reaches one.
      Choreography.boundingRadius(segment.dimension, Choreography.extrusion(segment, u))
      Frame(label, spec, reachOf(spec))

  private lazy val framesByTop: Map[Int, Seq[Frame]] = tops.map(top => top -> framesOf(top)).toMap

  private def plinthTopOf(top: Int): Float = plinthTop(DimensionLadder.sceneAt(0f, cycle, top))

  "The rest height" should "keep every frame's lowest point above the plinth's clearance" in:
    for top <- tops; frame <- framesByTop(top) do
      withClue(s"top $top, ${frame.label}: "):
        frame.spec.y - frame.reach should be >= plinthTopOf(top) + clearance - noise

  it should "just touch the clearance at the lowest frame of the cycle" in:
    for top <- tops do
      val lowest = framesByTop(top).map(f => f.spec.y - f.reach).min
      withClue(s"top $top: ")(lowest shouldBe plinthTopOf(top) + clearance +- touching)

  it should "match the stored depth below the centre, recomputed over every frame" in:
    for top <- tops do
      withClue(s"top $top: ")(framesByTop(top).map(_.reach).max shouldBe Stage.restDepth(top) +-
        touching)

  it should "hold the object's centre still at every t" in:
    for top <- tops do
      val centres = framesByTop(top).map(f => (f.spec.x, f.spec.y, f.spec.z)).distinct
      withClue(s"top $top: ")(centres should have length 1)

  it should "lift the tube radius into the rest depth, never below the default look's" in:
    val thick = Look.default.copy(edgeRadius = Look.default.edgeRadius + 0.02f)
    val thin = Look.default.copy(edgeRadius = 0.001f)
    def centreY(look: Look) = DimensionLadder.sceneAt(0f, look = look).objects.last.toObjectSpec.y
    centreY(thick) shouldBe centreY(Look.default) + 0.02f +- noise
    centreY(thin) shouldBe centreY(Look.default) +- noise

  // The corners of the n-cube are the lowest points of the sponge in it: checked on the sponge's
  // own mesh, at the breath of every dimension.
  private def spongeLevelsFor(n: Int): List[Float] = if n <= 4 then List(1f, 2f) else List(1f)

  it should "find the lowest point of a sponge at its n-cube's lowest corner" in:
    val timeline = DimensionLadder.segments(cycle, 8).filter(_.kind == Kind.Breath)
    for breath <- timeline; u <- List(0.25f, 0.5f); level <- spongeLevelsFor(breath.dimension) do
      val spec = DimensionLadder.objectAt(breath, u, cycle, 8).toObjectSpec
      val (hull, sponge) = (hullOf(spec), spec.copy(level = Some(level)))
      withClue(s"${breath.dimension}D, u $u, level $level: "):
        WorldMesh.corners(sponge).size should be > WorldMesh.corners(hull).size
        lowestY(sponge) should be >= lowestY(hull) - noise
        lowestY(sponge) shouldBe lowestY(hull) +- WorldMesh.vertexTolerance
