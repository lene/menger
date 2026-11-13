package examples.dsl

import examples.dsl.DimensionLadder.Kind
import examples.dsl.DimensionLadder.Segment
import menger.common.Material
import menger.common.ProfilingConfig
import menger.dsl.Camera
import menger.dsl.Cylinder
import menger.dsl.ResourceLimits
import menger.dsl.Scene
import menger.dsl.SceneObject
import menger.dsl.Sun
import menger.dsl.Vec3
import menger.engines.scene.MeshFactory
import menger.engines.scene.SceneCost
import menger.engines.scene.TriangleMeshSceneBuilder
import menger.objects.higher_d.NdSpongeVariant
import menger.spec.ObjectSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.io.Source
import scala.util.Using

class DimensionLadderSuite extends AnyFlatSpec with Matchers:

  private val cycle = DimensionLadder.cycleSeconds
  private val tops = 4 to 8
  private val vertexTolerance = 1e-3f
  private val degenerateArea = 1e-9f
  private val budgetFrames = 400
  private val aroundBoundary = 1e-3f
  private val cameraTolerance = 5e-3f
  private val materialTolerance = 1e-5f
  private val saneRadius = 5f
  // DimensionLadder's clearance and edge tube radius: the object rests that far above its bound.
  private val clearance = 0.01f
  private val edgeTube = 0.012f
  // How far above the plinth the lowest point may hover, as a fraction of the current object's
  // bounding radius. The rest height is that radius, which a turned object reaches only at its
  // worst angle; over tops 4 to 8 and 400 frames the measured worst is 0.67 of it (top 8).
  private val looseness = 0.7f
  // The renderer's field of view is 45 degrees horizontal; the video is 16:9 (menger#160).
  private val tanHalfFov = math.tan(math.toRadians(22.5)).toFloat
  private val frameAspect = 16f / 9f
  private val framingMargin = 0.95f
  private val framingTolerance = 1e-3f

  private type Point = (Float, Float, Float)

  /** The corners of the object's non-degenerate triangles, in world coordinates. */
  private def worldCorners(spec: ObjectSpec): Iterator[Point] =
    given ProfilingConfig = ProfilingConfig.disabled
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
    mesh.indices.grouped(3).map(_.map(at)).filter(t => area(t(0), t(1), t(2)) > degenerateArea)
      .flatten

  /** The distinct corners of the object's non-degenerate triangles, in world coordinates. */
  private def worldVertices(obj: SceneObject): Seq[Point] =
    worldCorners(obj.toObjectSpec).foldLeft(List.empty[Point]): (kept, p) =>
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

  private def materialValues(m: Option[Material]): List[Float] =
    m.toList.flatMap(x => List(x.color.r, x.color.g, x.color.b, x.color.a, x.ior, x.roughness,
      x.metallic, x.specular, x.emission, x.filmThickness, x.dispersion))

  private def withinTolerance(a: List[Float], b: List[Float]): Unit =
    a.length shouldBe b.length
    a.zip(b).foreach((x, y) => x shouldBe y +- materialTolerance)

  private def withinBudget(scene: Scene): Unit =
    val caustics = scene.caustics.fold(SceneCost.CausticsLoad.None)(c =>
      SceneCost.CausticsLoad(c.photonsPerIteration.toLong, c.iterations.toLong))
    val estimate = SceneCost.estimate(scene.objects.map(_.toObjectSpec),
      scene.render.fold(1)(_.accumulation), caustics)
    SceneCost.check(estimate) shouldBe Right(())

  "DimensionLadder" should "close the seam: scene(0) == scene(T)" in:
    DimensionLadder.scene(0f) shouldBe DimensionLadder.scene(cycle)
    DimensionLadder.scene(-cycle) shouldBe DimensionLadder.scene(0f)
    DimensionLadder.scene(0f).camera shouldBe DimensionLadder.scene(cycle).camera

  it should "be periodic: scene(t) == scene(t + T)" in:
    val t = cycle / 3f
    DimensionLadder.scene(t) shouldBe DimensionLadder.scene(t + cycle)

  it should "have segment durations that sum to the cycle length, for every cycle" in:
    for length <- List(60f, 180f, 600f); top <- tops do
      val all = DimensionLadder.segments(length, top)
      all.map(_.seconds).sum shouldBe length +- 1e-3f
      all.last.end shouldBe length +- 1e-3f
      all.zip(all.tail).foreach((a, b) => b.start shouldBe a.end)
      all.filter(_.kind == Kind.Collapse).map(_.seconds).sum shouldBe
        length * DimensionLadder.descentFraction +- 1e-3f

  it should "reject a cycle that is not positive, and clamp how far into a segment it is" in:
    an[IllegalArgumentException] should be thrownBy DimensionLadder.segments(0f)
    an[IllegalArgumentException] should be thrownBy DimensionLadder.phaseOf(1f, -1f)
    DimensionLadder.segmentAt(-1f)._2 shouldBe 0f
    DimensionLadder.segmentAt(cycle + 1f)._2 shouldBe 1f

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

  it should "show the lower object while a fold starts or a collapse ends, by flatness" in:
    def kind(spec: menger.spec.ObjectSpec): (String, Option[Int]) =
      (spec.objectType, spec.dimension)
    def expected(n: Int): (String, Option[Int]) =
      if n == 3 then ("cube", None) else ("n-cube", Some(n))
    for top <- tops; segment <- DimensionLadder.segments(cycle, top) do
      val n = segment.dimension
      def shown(u: Float) = kind(DimensionLadder.objectAt(segment, u, cycle, top).toObjectSpec)
      segment.kind match
        case Kind.Fold =>
          withClue(s"top $top, $segment: "):
            List(0f, 0.01f).foreach(shown(_) shouldBe expected(n - 1))
            shown(0.5f) shouldBe expected(n)
        case Kind.Collapse =>
          withClue(s"top $top, $segment: "):
            List(1f, 0.99f).foreach(shown(_) shouldBe expected(n - 1))
            shown(0.5f) shouldBe expected(n)
        case _ => ()

  it should "agree on material and edges across every segment boundary" in:
    for top <- tops; (before, after) <- boundaries(top) do
      withClue(s"top $top, $before -> $after: "):
        val left = DimensionLadder.objectAt(before, 1f, cycle, top).toObjectSpec
        val right = DimensionLadder.objectAt(after, 0f, cycle, top).toObjectSpec
        withinTolerance(materialValues(left.material), materialValues(right.material))
        withinTolerance(left.edgeRadius.toList, right.edgeRadius.toList)
        withinTolerance(materialValues(left.edgeMaterial), materialValues(right.edgeMaterial))

  it should "move the camera continuously across every segment boundary" in:
    for top <- tops; segment <- DimensionLadder.segments(cycle, top) do
      val at = segment.start
      val (before, after) = (DimensionLadder.sceneAt(at - aroundBoundary, cycle, top).camera,
        DimensionLadder.sceneAt(at + aroundBoundary, cycle, top).camera)
      withClue(s"top $top, $segment: "):
        List(before.position.x - after.position.x, before.position.y - after.position.y,
          before.position.z - after.position.z).foreach(_ shouldBe 0f +- cameraTolerance)

  it should "move the object and the aim of the camera continuously across every boundary" in:
    def centreAndAim(t: Float, top: Int): List[Float] =
      val scene = DimensionLadder.sceneAt(t, cycle, top)
      val spec = scene.objects.last.toObjectSpec
      List(spec.x, spec.y, spec.z, scene.camera.lookAt.x, scene.camera.lookAt.y,
        scene.camera.lookAt.z)
    for top <- tops; segment <- DimensionLadder.segments(cycle, top) do
      withClue(s"top $top, $segment: "):
        centreAndAim(segment.start - aroundBoundary, top)
          .zip(centreAndAim(segment.start + aroundBoundary, top))
          .foreach((before, after) => before shouldBe after +- cameraTolerance)

  private def plinthOf(scene: Scene): Cylinder =
    scene.objects.collectFirst { case c: Cylinder => c }.get

  /** The bounding radius the object of `scene` rests on: its centre less the plinth's top, the
    * clearance and the edge tube. */
  private def restRadius(scene: Scene): Float =
    scene.objects.last.toObjectSpec.y - plinthOf(scene).p1.get.y - clearance - edgeTube

  it should "rest its lowest point above the plinth, and close to it, in every frame" in:
    for top <- tops; i <- 0 until budgetFrames do
      val scene = DimensionLadder.sceneAt(cycle * i / budgetFrames, cycle, top)
      // A sponge's lowest point is its hull's, and its level-0 mesh is the hull: cheap to build.
      val spec = scene.objects.last.toObjectSpec
      val lowest = worldCorners(spec.copy(level = spec.level.map(_ => 0f))).map(_._2).min -
        plinthOf(scene).p1.get.y
      withClue(s"top $top, frame $i: "):
        lowest should be >= clearance - vertexTolerance
        lowest should be <= looseness * restRadius(scene)

  /** The point as fractions of the frame's half width and half height, seen from `camera`. */
  private def inFrame(camera: Camera, point: Vec3): (Float, Float) =
    val view = (camera.lookAt - camera.position).normalize
    val right = view.cross(camera.up).normalize
    val offset = point - camera.position
    val depth = offset.dot(view)
    depth should be > 0f
    (offset.dot(right) / depth / tanHalfFov,
      offset.dot(right.cross(view)) / depth / tanHalfFov * frameAspect)

  it should "frame the object, the plinth's front edge and its width at 16:9, in every frame" in:
    for top <- tops; i <- 0 until budgetFrames do
      val scene = DimensionLadder.sceneAt(cycle * i / budgetFrames, cycle, top)
      val (camera, plinth) = (scene.camera, plinthOf(scene))
      val centre = scene.objects.last.toObjectSpec
      val radius = plinth.radius.get
      val (bottom, topY) = (plinth.p0.get.y, plinth.p1.get.y)
      val away = Vec3(camera.position.x, 0f, camera.position.z)
      val (distance, toCamera) = (away.magnitude, away.normalize)
      val sideways = Vec3(-toCamera.z, 0f, toCamera.x)
      // The silhouette's tangent points, not the points at the axis' height across the view.
      val along = toCamera * (radius * radius / distance)
      val halfChord = radius * math.sqrt(1 - radius * radius / (distance * distance)).toFloat
      val across = sideways * halfChord
      val points = Vec3(centre.x, centre.y + restRadius(scene), centre.z) ::
        (toCamera * radius + Vec3(0f, bottom, 0f)) ::
        (for side <- List(along + across, along - across); y <- List(bottom, topY)
          yield side + Vec3(0f, y, 0f))
      points.foreach: point =>
        withClue(s"top $top, frame $i, point $point: "):
          val (x, y) = inFrame(camera, point)
          math.abs(x) should be <= framingMargin + framingTolerance
          math.abs(y) should be <= framingMargin + framingTolerance

  it should "turn every plane containing a collapsing axis a multiple of 180 degrees" in:
    for
      top <- tops
      collapse <- DimensionLadder.segments(cycle, top).filter(_.kind == Kind.Collapse)
    do
      val axis = DimensionLadder.axisLetters(collapse.dimension - 1)
      val containing = DimensionLadder.planeAngles(collapse.end, cycle, top)
        .filter(_.plane.letters.contains(axis))
      containing should not be empty
      containing.foreach: rotation =>
        withClue(s"top $top, plane ${rotation.plane}: "):
          val offset = rotation.angle.value % 180f
          math.min(offset, 180f - offset) shouldBe 0f +- 1e-3f

  it should "descend in planes among the extra axes only" in:
    val descent = DimensionLadder.turns(cycle, 6)
      .filter(_.from >= cycle * (1f - DimensionLadder.descentFraction) - 1e-3f)
    descent.map(_.plane).toSet shouldBe Set("wu", "vu", "wv")
    descent.foreach: turn =>
      turn.plane.foreach(letter => DimensionLadder.axisLetters.drop(3) should contain(letter))

  it should "keep the projected half extent finite and inside the bounding radius, 3D to 8D" in:
    for n <- 3 to 8; last <- List(0f, 0.5f, 1f) do
      val extent = DimensionLadder.halfExtent(n, last)
      withClue(s"dimension $n, extrusion $last: "):
        extent.isFinite shouldBe true
        extent should be < saneRadius
        extent should be <= DimensionLadder.boundingRadius(n) + materialTolerance
        DimensionLadder.boundingRadius(n) should be < saneRadius

  it should "bound the half extent by a radius that is the lower cube's where an axis is flat" in:
    for n <- 4 to 8 do
      DimensionLadder.boundingRadius(n, 0f) shouldBe DimensionLadder.boundingRadius(n - 1) +-
        materialTolerance
      for last <- List(0f, 0.25f, 0.5f, 0.75f, 1f) do
        withClue(s"dimension $n, extrusion $last: "):
          DimensionLadder.halfExtent(n, last) should be <=
            DimensionLadder.boundingRadius(n, last) + materialTolerance

  it should "pass the scene budget at every sampled frame and breath peak, for every top" in:
    for top <- tops do
      val peaks = DimensionLadder.segments(cycle, top).filter(_.kind == Kind.Breath)
        .map(s => s.start + s.seconds / 2f)
      val sampled = (0 until budgetFrames).map(i => cycle * i / budgetFrames)
      (sampled ++ peaks).foreach: t =>
        withClue(s"top $top, t $t: ")(withinBudget(DimensionLadder.sceneAt(t, cycle, top)))

  it should "hold the seam and the timeline for every top dimension" in:
    for top <- tops do
      DimensionLadder.sceneAt(0f, cycle, top) shouldBe DimensionLadder.sceneAt(cycle, cycle, top)
      DimensionLadder.segments(cycle, top).count(_.kind == Kind.Fold) shouldBe top - 3
      DimensionLadder.segments(cycle, top).count(_.kind == Kind.Collapse) shouldBe top - 3

  it should "breathe a surface sponge above 3D, at the level the bell gives" in:
    for top <- tops; breath <- DimensionLadder.segments(cycle, top).filter(_.kind == Kind.Breath) do
      val n = breath.dimension
      val spec = DimensionLadder.objectAt(breath, 0.25f, cycle, top).toObjectSpec
      withClue(s"top $top, $breath: "):
        if n == 3 then spec.objectType shouldBe "sponge-volume"
        else
          spec.objectType shouldBe "n-sponge-surface"
          spec.dimension shouldBe Some(n)
          spec.level.get shouldBe DimensionLadder.breathDepth(n) * 0.5f +- 1e-4f

  it should "keep every breath depth within the surface sponge's budget limit" in:
    for n <- 4 to 8 do
      DimensionLadder.breathDepth(n) should be <=
        ResourceLimits.ndSpongeLevel(NdSpongeVariant.Surface, n).max
      DimensionLadder.breathDepth(n) should be >= 1

  it should "render with accumulation and without adaptive antialiasing at any t" in:
    for t <- List(0f, 7.3f, cycle / 2f, cycle - 0.1f); top <- tops do
      val settings = DimensionLadder.sceneAt(t, cycle, top).render.get
      settings.antialiasing shouldBe false
      settings.accumulation shouldBe 16

  private val scriptsDir = List("scripts", "../scripts", "../../scripts")
    .find(d => java.io.File(d, "integration-tests.sh").exists()).get

  private def script(name: String): String =
    Using.resource(Source.fromFile(s"$scriptsDir/$name"))(_.mkString)

  // The freeze frames, as named in integration-tests.sh, the one place their t values are
  // defined: the segment each is in, and whether it is the middle of it or the start.
  private val freezeFrames = List(
    ("cube", Kind.Hold, 3, 0f), ("3D breath", Kind.Breath, 3, 0.5f),
    ("fold 4D", Kind.Fold, 4, 0.5f), ("breath 4D", Kind.Breath, 4, 0.5f),
    ("fold 5D", Kind.Fold, 5, 0.5f), ("breath 6D", Kind.Breath, 6, 0.5f),
    ("descent", Kind.Collapse, 5, 0.5f))

  it should "put every freeze frame of the scripts in the segment its name says" in:
    val frame = """DimensionLadder ([^"]+)"[^\n]*\n\s*require_asset[^\n]*--t ([0-9.]+)""".r
    val defined = frame.findAllMatchIn(script("integration-tests.sh"))
      .map(m => (m.group(1), m.group(2).toFloat)).toList
    defined.map(_._1) shouldBe freezeFrames.map(_._1)
    defined.zip(freezeFrames).foreach:
      case ((name, t), (_, kind, dimension, middle)) =>
        val (segment, u) = DimensionLadder.segmentAt(t, cycle, DimensionLadder.topDimension)
        withClue(s"$name, t $t: "):
          (segment.kind, segment.dimension) shouldBe (kind, dimension)
          u shouldBe middle +- 1e-3f
    val manual = """examples\.dsl\.DimensionLadder --t ([0-9.]+)""".r
      .findAllMatchIn(script("manual-test.sh")).map(_.group(1).toFloat).toList
    manual shouldBe defined.map(_._2)

  it should "light the scene with the sun first, because only the first light emits caustics" in:
    DimensionLadder.scene(10f).lights.head shouldBe a[Sun]
