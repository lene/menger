package examples.dsl

import examples.dsl.DimensionLadder.Kind
import examples.dsl.DimensionLadder.Segment
import menger.common.ProfilingConfig
import menger.dsl.Scene
import menger.dsl.SceneObject
import menger.dsl.Sun
import menger.engines.scene.MeshFactory
import menger.engines.scene.SceneCost
import menger.engines.scene.TriangleMeshSceneBuilder
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DimensionLadderSuite extends AnyFlatSpec with Matchers:

  private val cycle = DimensionLadder.cycleSeconds
  private val tops = 4 to 8
  private val vertexTolerance = 1e-3f
  private val degenerateArea = 1e-9f
  private val frames = 200

  private type Point = (Float, Float, Float)

  /** The distinct corners of the object's non-degenerate triangles, in world coordinates. */
  private def worldVertices(obj: SceneObject): Seq[Point] =
    given ProfilingConfig = ProfilingConfig.disabled
    val spec = obj.toObjectSpec
    val mesh = MeshFactory.create(spec)
    val m = TriangleMeshSceneBuilder.instanceTransform(spec)
    def at(i: Int): Point =
      val base = i * mesh.vertexStride
      val (x, y, z) = (mesh.vertices(base), mesh.vertices(base + 1), mesh.vertices(base + 2))
      (m(0) * x + m(1) * y + m(2) * z + m(3), m(4) * x + m(5) * y + m(6) * z + m(7),
        m(8) * x + m(9) * y + m(10) * z + m(11))
    def area(a: Point, b: Point, c: Point): Float =
      val (ux, uy, uz) = (b._1 - a._1, b._2 - a._2, b._3 - a._3)
      val (vx, vy, vz) = (c._1 - a._1, c._2 - a._2, c._3 - a._3)
      val (cx, cy, cz) = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
      0.5f * math.sqrt(cx * cx + cy * cy + cz * cz).toFloat
    val corners = mesh.indices.grouped(3).map(_.map(at))
      .filter(t => area(t(0), t(1), t(2)) > degenerateArea)
    corners.flatten.foldLeft(List.empty[Point]): (kept, p) =>
      if kept.exists(q => near(p, q)) then kept else p :: kept

  private def near(p: Point, q: Point): Boolean =
    math.abs(p._1 - q._1) < vertexTolerance && math.abs(p._2 - q._2) < vertexTolerance &&
      math.abs(p._3 - q._3) < vertexTolerance

  private def sameShape(a: SceneObject, b: SceneObject): Unit =
    val (left, right) = (worldVertices(a), worldVertices(b))
    left should not be empty
    left.filterNot(p => right.exists(near(p, _))) shouldBe empty
    right.filterNot(p => left.exists(near(p, _))) shouldBe empty

  private def boundaries(top: Int): List[(Segment, Segment)] =
    val all = DimensionLadder.segments(cycle, top)
    all.zip(all.tail :+ all.head)

  "DimensionLadder" should "close the seam: scene(0) == scene(T)" in:
    DimensionLadder.scene(0f) shouldBe DimensionLadder.scene(cycle)
    DimensionLadder.scene(-cycle) shouldBe DimensionLadder.scene(0f)

  it should "be periodic: scene(t) == scene(t + T)" in:
    val t = cycle / 3f
    DimensionLadder.scene(t) shouldBe DimensionLadder.scene(t + cycle)

  it should "have segment durations that sum to the cycle length, for every cycle" in:
    for length <- List(60f, 180f, 600f); top <- tops do
      val all = DimensionLadder.segments(length, top)
      all.map(_.seconds).sum shouldBe length +- 1e-3f
      all.last.end shouldBe length +- 1e-3f
      all.zip(all.tail).foreach((a, b) => b.start shouldBe a.end)
      all.filter(_.kind == Kind.Collapse).map(_.seconds).sum shouldBe length / 5f +- 1e-3f

  it should "hand over every segment boundary with the same projected vertices" in:
    for top <- tops; (before, after) <- boundaries(top) do
      withClue(s"top $top, $before -> $after: "):
        sameShape(DimensionLadder.objectAt(before, 1f, cycle, top),
          DimensionLadder.objectAt(after, 0f, cycle, top))

  it should "keep the higher object the lower one's shape just after a fold starts" in:
    val justAfter = 1e-5f
    for top <- tops; fold <- DimensionLadder.segments(cycle, top).filter(_.kind == Kind.Fold) do
      withClue(s"top $top, $fold: "):
        sameShape(DimensionLadder.objectAt(fold, 0f, cycle, top),
          DimensionLadder.objectAt(fold, justAfter, cycle, top))

  it should "turn every plane containing a collapsing axis a multiple of 180 degrees" in:
    for
      top <- tops
      collapse <- DimensionLadder.segments(cycle, top).filter(_.kind == Kind.Collapse)
    do
      val axis = "xyzwvuts"(collapse.dimension - 1)
      val containing = DimensionLadder.planeAngles(collapse.end, cycle, top)
        .filter(_.plane.letters.contains(axis))
      containing should not be empty
      containing.foreach: rotation =>
        withClue(s"top $top, plane ${rotation.plane}: "):
          val offset = rotation.angle.value % 180f
          math.min(offset, 180f - offset) shouldBe 0f +- 1e-3f

  it should "descend in planes among the extra axes only" in:
    val descent = DimensionLadder.turns(cycle, 6).filter(_.from >= cycle * 0.8f - 1e-3f)
    descent.map(_.plane).toSet shouldBe Set("wu", "vu", "wv")
    descent.foreach: turn =>
      turn.plane.foreach(letter => "wvuts" should contain(letter))

  it should "pass the scene budget at every sampled frame" in:
    for i <- 0 until frames do
      val scene: Scene = DimensionLadder.scene(cycle * i / frames)
      val caustics = scene.caustics.fold(SceneCost.CausticsLoad.None)(c =>
        SceneCost.CausticsLoad(c.photonsPerIteration.toLong, c.iterations.toLong))
      val estimate = SceneCost.estimate(scene.objects.map(_.toObjectSpec),
        scene.render.fold(1)(_.accumulation), caustics)
      withClue(s"frame $i: ")(SceneCost.check(estimate) shouldBe Right(()))

  it should "hold the seam and the timeline for every top dimension" in:
    for top <- tops do
      DimensionLadder.sceneAt(0f, cycle, top) shouldBe DimensionLadder.sceneAt(cycle, cycle, top)
      DimensionLadder.segments(cycle, top).count(_.kind == Kind.Fold) shouldBe top - 3
      DimensionLadder.segments(cycle, top).count(_.kind == Kind.Collapse) shouldBe top - 3

  it should "light the scene with the sun first, because only the first light emits caustics" in:
    DimensionLadder.scene(10f).lights.head shouldBe a[Sun]
