package examples.dsl.ladder

import examples.dsl.ladder.DimensionLadder.Kind
import examples.dsl.ladder.DimensionLadder.Segment
import menger.common.Material
import menger.dsl.Camera
import menger.dsl.Cylinder
import menger.dsl.ResourceLimits
import menger.dsl.Scene
import menger.dsl.SceneObject
import menger.dsl.Sun
import menger.dsl.Vec3
import menger.engines.scene.SceneCost
import menger.objects.higher_d.NdSpongeVariant
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.io.Source
import scala.util.Using

class DimensionLadderSuite extends AnyFlatSpec with Matchers:

  private val cycle = DimensionLadder.cycleSeconds
  private val tops = 4 to 8
  private val budgetFrames = 400
  private val aroundBoundary = 1e-3f
  private val cameraTolerance = 5e-3f
  private val materialTolerance = 1e-5f
  private val saneRadius = 5f
  // The renderer's field of view is 45 degrees horizontal; the video is 16:9 (menger#160).
  private val tanHalfFov = math.tan(math.toRadians(22.5)).toFloat
  private val frameAspect = 16f / 9f
  private val framingMargin = 0.95f
  private val framingTolerance = 1e-3f

  private def sameShape(a: SceneObject, b: SceneObject): Unit =
    val (left, right) = (WorldMesh.vertices(a), WorldMesh.vertices(b))
    left should not be empty
    left.filterNot(p => right.exists(WorldMesh.near(p, _))) shouldBe empty
    right.filterNot(p => left.exists(WorldMesh.near(p, _))) shouldBe empty

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
        length * DimensionLadder.descentFraction(top) +- 1e-3f

  it should "keep the old pace: the ascent at 36/19 s per weight unit, a descent of 36 s" in:
    val pace = 36f / 19f
    cycle shouldBe 88f * pace + 36f +- 1e-3f
    val all = DimensionLadder.segments(cycle, 6)
    all.filter(_.kind == Kind.Collapse).map(_.seconds).sum shouldBe 36f +- 1e-3f
    all.find(_.kind == Kind.Hold).get.seconds shouldBe 2f * pace +- 1e-3f
    val breath = all.find(s => s.kind == Kind.Breath && s.dimension == 3).get
    breath.start shouldBe 3.789f +- 1e-3f
    breath.end shouldBe 41.684f +- 1e-3f
    breath.seconds shouldBe 20f * pace +- 1e-3f

  it should "lengthen the cycle with deeper breaths instead of squeezing the segments" in:
    Choreography.naturalCycle(6, Map(3 -> 5, 4 -> 4, 5 -> 3, 6 -> 2)) should be >
      Choreography.naturalCycle(6, Map(3 -> 5, 4 -> 3, 5 -> 2, 6 -> 1))
    Choreography.naturalCycle(6, Map(3 -> 5, 4 -> 3, 5 -> 2, 6 -> 1)) shouldBe 180f +- 1e-3f

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
      val t = cycle * i / budgetFrames
      val scene = DimensionLadder.sceneAt(t, cycle, top)
      val (camera, plinth) = (scene.camera, plinthOf(scene))
      val centre = scene.objects.last.toObjectSpec
      val (segment, u) = DimensionLadder.segmentAt(t, cycle, top)
      val objectRadius =
        DimensionLadder.boundingRadius(segment.dimension, Choreography.extrusion(segment, u))
      val radius = plinth.radius.get
      val (bottom, topY) = (plinth.p0.get.y, plinth.p1.get.y)
      val away = Vec3(camera.position.x, 0f, camera.position.z)
      val (distance, toCamera) = (away.magnitude, away.normalize)
      val sideways = Vec3(-toCamera.z, 0f, toCamera.x)
      // The silhouette's tangent points, not the points at the axis' height across the view.
      val along = toCamera * (radius * radius / distance)
      val halfChord = radius * math.sqrt(1 - radius * radius / (distance * distance)).toFloat
      val across = sideways * halfChord
      val points = Vec3(centre.x, centre.y + objectRadius, centre.z) ::
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
      .filter(_.from >= cycle * (1f - DimensionLadder.descentFraction(6)) - 1e-3f)
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
    val manual = """examples\.dsl\.ladder\.DimensionLadder --t ([0-9.]+)""".r
      .findAllMatchIn(script("manual-test.sh")).map(_.group(1).toFloat).toList
    manual shouldBe defined.map(_._2)

  it should "light the scene with the sun first, because only the first light emits caustics" in:
    DimensionLadder.scene(10f).lights.head shouldBe a[Sun]

  private val chrome = Look.default.copy(metal = menger.dsl.Material.Chrome)

  "A look" should "change materials and nothing else" in:
    for t <- List(0f, 22.7f, 51.2f, 72f, 162f) do
      val (plain, changed) = (DimensionLadder.sceneAt(t), DimensionLadder.sceneAt(t, look = chrome))
      withClue(s"t $t: "):
        changed.camera shouldBe plain.camera
        val (before, after) = (plain.objects.last.toObjectSpec, changed.objects.last.toObjectSpec)
        after.objectType shouldBe before.objectType
        after.level shouldBe before.level
        after.size shouldBe before.size
        after.projection shouldBe before.projection
        after.scale shouldBe before.scale
    DimensionLadder.sceneAt(0f, look = chrome).objects.last.toObjectSpec.material shouldBe
      Some(menger.dsl.Material.Chrome.toCoreMaterial)

  it should "set the breath depths" in:
    val look = Look.default.copy(depths = Map(3 -> 2))
    val breath = Choreography(cycle, DimensionLadder.topDimension, look.depths).segments
      .find(s => s.kind == Kind.Breath && s.dimension == 3).get
    val spec = DimensionLadder.sceneAt(breath.start + breath.seconds / 2f, look = look)
      .objects.last.toObjectSpec
    spec.level.get shouldBe 2f +- 1e-3f

  it should "add fill lights after the sun" in:
    val fill = menger.dsl.Directional(Vec3(0f, -1f, 0f))
    val lights = DimensionLadder.sceneAt(10f, look = Look.default.copy(fillLights = List(fill)))
      .lights
    lights.head shouldBe a[Sun]
    lights should have length 2

  "Choreography" should "be equal for equal arguments" in:
    val (first, second) = (Choreography(cycle, 6, Look.default.depths),
      Choreography(cycle, 6, Look.default.depths))
    first shouldBe second
    first.segments shouldBe second.segments

  // The default look's scene at the six times the freeze-frame references and the experiments
  // look at, with the cameras the ladder had before the look got its framing, turns, depth and
  // crossfade fields (the pre-change scenes, to the last bit).
  private val sixTimes = List(0f, 22.7f, 51.2f, 72f, 102f, 162f)
  private val cameraBeforeFramingKnob = List(
    Vec3(-2.9823086f, 1.2204607f, -6.395582f), Vec3(-2.9823086f, 1.2204607f, -6.395582f),
    Vec3(-3.098141f, 1.2589805f, -6.643985f), Vec3(-3.3042817f, 1.327532f, -7.0860553f),
    Vec3(-3.5176127f, 1.3984749f, -7.5435452f), Vec3(-4.0993857f, 1.5919425f, -8.791162f))

  private def objectOf(scene: Scene): SceneObject = scene.objects.last

  /** What a look may change without moving the camera or the object's shape: its type, level
    * and size. */
  private def sameShapeAndFrame(changed: Scene, plain: Scene): Unit =
    changed.camera shouldBe plain.camera
    val (after, before) = (objectOf(changed).toObjectSpec, objectOf(plain).toObjectSpec)
    after.objectType shouldBe before.objectType
    after.level shouldBe before.level
    after.size shouldBe before.size
    after.projection shouldBe before.projection
    after.scale shouldBe before.scale

  "The default look" should "build the scene it built before the new fields, at six times" in:
    val fresh = Look.default.copy()
    for (t, camera) <- sixTimes.zip(cameraBeforeFramingKnob) do
      withClue(s"t $t: "):
        DimensionLadder.sceneAt(t, look = fresh) shouldBe DimensionLadder.sceneAt(t)
        DimensionLadder.sceneAt(t, look = fresh) shouldBe DimensionLadder.scene(t)
        DimensionLadder.sceneAt(t).camera.position shouldBe camera
        DimensionLadder.sceneAt(t).camera.lookAt shouldBe Vec3(0f, 0.22869998f, 0f)

  // The object of the scene at each of the six times, as `toString` printed it at base commit
  // 0c728092, before the look had these fields: its type, level, material, size, rotation,
  // projection, scale and edge tubes, to the last digit. Captured from that commit's code.
  private val objectBeforeNewFields = Map(
    0f -> (
      "Cube(Vec3(0.0,0.90739995,0.0),Some(Material(Color(0.78,0.82,0.95,1.0),1.0,0.1,1.0,1." +
      "0,0.0,0.0,0.0)),None,1.0,1.0,None,None,None,None,0,1.0,None,None,Vec3(0.0,0.0,0.0))"),
    22.7f -> (
      "Sponge(VolumeFilling,Vec3(0.0,0.90739995,0.0),4.9999533,Some(Material(Color(0.999997" +
      "9,0.99999833,0.9999995,0.020009171),1.4999952,9.357929E-7,9.357929E-6,1.0,0.0,0.0,0." +
      "0)),None,1.0,1.0,None,None,None,None,0,1.0,None,None,Vec3(0.0,1.4070288,0.0))"),
    51.2f -> (
      "Hypercube(4,Vec3(0.0,0.90739995,0.0),Some(Material(Color(1.0,1.0,1.0,0.2),1.0,0.1,0." +
      "0,0.5,0.0,500.0,0.0)),None,1.25,1.0,None,None,None,None,0,1.0,Some(ProjectionSpec(4," +
      "List(PlaneRotation(xw,6.7308326), PlaneRotation(yw,6.7308326), PlaneRotation(zw,6.73" +
      "08326), PlaneRotation(xz,181.83176)),3.0,1.5,List(ProjectionStage(5.0,1.0)))),Some(0" +
      ".012),Some(Material(Color(0.78,0.82,0.95,1.0),1.0,0.1,1.0,1.0,0.0,0.0,0.0)),Vec3(0.0" +
      ",0.0,0.0),Some(Scale<1, 1, 1, 0.51>))"),
    72f -> (
      "HypercubeSponge(4,SurfaceSubdividing,3.414212,1,Vec3(0.0,0.90739995,0.0),Some(Materi" +
      "al(Color(0.96778166,0.97363955,0.9926776,0.16351806),1.4267764,0.0146447,0.146447,1." +
      "0,0.0,0.0,0.0)),None,1.25,1.0,None,None,None,None,0,1.0,Some(ProjectionSpec(4,List(P" +
      "laneRotation(xw,100.3641), PlaneRotation(yw,100.3641), PlaneRotation(zw,100.3641), P" +
      "laneRotation(xz,255.70093)),3.0,1.5,List(ProjectionStage(5.0,1.0)))),None,Vec3(0.0,0" +
      ".0,0.0))"),
    102f -> (
      "Hypercube(5,Vec3(0.0,0.90739995,0.0),Some(Material(Color(1.0,1.0,1.0,0.2),1.0,0.1,0." +
      "0,0.5,0.0,500.0,0.0)),None,1.5625,1.0,None,None,None,None,0,1.0,Some(ProjectionSpec(" +
      "5,List(PlaneRotation(xw,180.0), PlaneRotation(yw,180.0), PlaneRotation(zw,180.0), Pl" +
      "aneRotation(xv,14.490178), PlaneRotation(yv,14.490178), PlaneRotation(zv,14.490178)," +
      " PlaneRotation(wv,0.0), PlaneRotation(xz,362.24298)),3.0,1.5,List(ProjectionStage(6." +
      "25,1.25), ProjectionStage(5.0,1.0)))),Some(0.012),Some(Material(Color(0.78,0.82,0.95" +
      ",1.0),1.0,0.1,1.0,1.0,0.0,0.0,0.0)),Vec3(0.0,0.0,0.0),Some(Scale<1, 1, 1, 1, 0.70>))"),
    162f -> (
      "HypercubeSponge(6,SurfaceSubdividing,1.38269,1,Vec3(0.0,0.90739995,0.0),Some(Materia" +
      "l(Color(0.9320959,0.9444421,0.9845673,0.32248193),1.3456725,0.030865503,0.30865502,1" +
      ".0,0.0,0.0,0.0)),None,1.953125,1.0,None,None,None,None,0,1.0,Some(ProjectionSpec(6,L" +
      "ist(PlaneRotation(xw,180.0), PlaneRotation(yw,180.0), PlaneRotation(zw,180.0), Plane" +
      "Rotation(xv,180.0), PlaneRotation(yv,180.0), PlaneRotation(zv,180.0), PlaneRotation(" +
      "xu,168.1896), PlaneRotation(yu,168.1896), PlaneRotation(zu,168.1896), PlaneRotation(" +
      "wu,0.0), PlaneRotation(vu,0.0), PlaneRotation(wv,0.0), PlaneRotation(xz,575.3271)),3" +
      ".0,1.5,List(ProjectionStage(7.8125,1.5625), ProjectionStage(6.25,1.25), ProjectionSt" +
      "age(5.0,1.0)))),None,Vec3(0.0,0.0,0.0))"))
  private val lightsBeforeNewFields =
    "List(Sun(Vec3(0.5,-0.7,0.5),0.53,1000.0,Color(1.0,1.0,1.0,1.0),1.0,None))"

  it should "build the object and the lights it built at base commit 0c728092" in:
    objectBeforeNewFields.keySet shouldBe sixTimes.toSet
    for (t, expected) <- objectBeforeNewFields do
      withClue(s"t $t: "):
        val scene = DimensionLadder.sceneAt(t)
        objectOf(scene).toString shouldBe expected
        scene.lights.toString shouldBe lightsBeforeNewFields
        scene.objects should have length 2

  it should "return the choreography's own turns, the turntable included" in:
    val timeline = Choreography(cycle, DimensionLadder.topDimension, Look.default.depths)
    for t <- sixTimes do
      Look.default.turns(timeline)(t) shouldBe timeline.planeAngles(t)

  private def breathOf3D(look: Look): Scene =
    val breath = Choreography(cycle, DimensionLadder.topDimension, look.depths).segments
      .find(s => s.kind == Kind.Breath && s.dimension == 3).get
    DimensionLadder.sceneAt(breath.start + breath.seconds / 4f, look = look)

  "Look.sponge3D" should "set the 3D breath's sponge type and nothing else" in:
    val ias = Look.default.copy(sponge3D = menger.dsl.RecursiveIAS)
    val (plain, changed) = (breathOf3D(Look.default), breathOf3D(ias))
    objectOf(plain).toObjectSpec.objectType shouldBe "sponge-volume"
    objectOf(changed).toObjectSpec.objectType shouldBe "sponge-recursive-ias"
    changed.camera shouldBe plain.camera
    objectOf(changed).toObjectSpec.level shouldBe objectOf(plain).toObjectSpec.level
    objectOf(changed).toObjectSpec.material shouldBe objectOf(plain).toObjectSpec.material

  it should "leave every other segment alone" in:
    val ias = Look.default.copy(sponge3D = menger.dsl.RecursiveIAS)
    for t <- sixTimes.filterNot(_ == 22.7f) do
      DimensionLadder.sceneAt(t, look = ias) shouldBe DimensionLadder.sceneAt(t)

  private val metalAlways: Look.Crossfade = (metal, _, _) => metal

  "Look.crossfade" should "set how metal and film mix while solidifying, and nothing else" in:
    val look = Look.default.copy(crossfade = metalAlways)
    val solidifying = DimensionLadder.segments(cycle).find(_.kind == Kind.Solidify).get
    val t = solidifying.start + solidifying.seconds / 2f
    val (plain, changed) = (DimensionLadder.sceneAt(t), DimensionLadder.sceneAt(t, look = look))
    sameShapeAndFrame(changed, plain)
    objectOf(changed).toObjectSpec.material shouldBe Some(look.metal.toCoreMaterial)
    objectOf(plain).toObjectSpec.material should not be Some(look.metal.toCoreMaterial)

  it should "not touch the breath's mix of metal and glass" in:
    val look = Look.default.copy(crossfade = metalAlways)
    breathOf3D(look) shouldBe breathOf3D(Look.default)

  "Look.breathMaterial" should "set the breath's material from its bell value, and nothing else" in:
    val look = Look.default.copy(breathMaterial = Some(_ => Look.default.metal))
    val (plain, changed) = (breathOf3D(Look.default), breathOf3D(look))
    sameShapeAndFrame(changed, plain)
    objectOf(changed).toObjectSpec.material shouldBe Some(Look.default.metal.toCoreMaterial)
    objectOf(plain).toObjectSpec.material should not be objectOf(changed).toObjectSpec.material

  it should "be given the bell value, 0 at the breath's ends and 1 at its middle" in:
    val seen = scala.collection.mutable.ListBuffer.empty[Float]
    val record: Float => menger.dsl.Material = b =>
      seen += b
      Look.default.metal
    val look = Look.default.copy(breathMaterial = Some(record))
    val breath = DimensionLadder.segments(cycle).find(s => s.kind == Kind.Breath).get
    List(0f, 0.5f).foreach: u =>
      DimensionLadder.objectAt(breath, u, look = look)
    seen.toList shouldBe List(0f, 1f)

  it should "leave the other segments alone" in:
    val look = Look.default.copy(breathMaterial = Some(_ => Look.default.glass))
    for t <- List(0f, 51.2f, 190f) do
      DimensionLadder.sceneAt(t, look = look) shouldBe DimensionLadder.sceneAt(t)

  /** `inFrame` for another field of view. */
  private def inFrameWith(halfTan: Float)(camera: Camera, point: Vec3): (Float, Float) =
    val view = (camera.lookAt - camera.position).normalize
    val right = view.cross(camera.up).normalize
    val offset = point - camera.position
    val depth = offset.dot(view)
    (offset.dot(right) / depth / halfTan,
      offset.dot(right.cross(view)) / depth / halfTan * frameAspect)

  "Look.framing" should "default to a 45 degree field, a margin of 0.95 and the bounding radius" in:
    Look.default.framing shouldBe Framing(fovDegrees = 45f, margin = 0.95f, fit = Fit.Bound)
    Framing.default.tanHalfFovX shouldBe 0.4142f

  it should "only move the camera, whatever it fits to" in:
    val extent = Look.default.copy(framing = Framing.default.copy(fit = Fit.Extent))
    for t <- sixTimes do
      withClue(s"t $t: "):
        val plain = DimensionLadder.sceneAt(t)
        val changed = DimensionLadder.sceneAt(t, look = extent)
        objectOf(changed) shouldBe objectOf(plain)
        changed.objects.head shouldBe plain.objects.head
        changed.lights shouldBe plain.lights
        changed.camera.lookAt shouldBe plain.camera.lookAt

  it should "fit the frame's own extent, which is never farther than the bound" in:
    val extent = Look.default.copy(framing = Framing.default.copy(fit = Fit.Extent))
    val distances = for
      t <- (0 until budgetFrames).map(cycle * _ / budgetFrames)
    yield (DimensionLadder.sceneAt(t).camera, DimensionLadder.sceneAt(t, look = extent).camera)
    def away(c: Camera) = (c.position - c.lookAt).magnitude
    distances.foreach((bound, tight) => away(tight) should be <= away(bound) + materialTolerance)
    distances.exists((bound, tight) => away(tight) < away(bound) - 1e-3f) shouldBe true

  it should "keep the object and the plinth inside the margin it is given" in:
    val framing = Framing(fovDegrees = 35f, margin = 0.85f, fit = Fit.Bound)
    val look = Look.default.copy(framing = framing)
    val halfTan = math.tan(math.toRadians(17.5)).toFloat
    for i <- 0 until budgetFrames do
      val t = cycle * i / budgetFrames
      val scene = DimensionLadder.sceneAt(t, look = look)
      val (segment, u) = DimensionLadder.segmentAt(t)
      val centre = objectOf(scene).toObjectSpec
      val radius =
        DimensionLadder.boundingRadius(segment.dimension, Choreography.extrusion(segment, u))
      val top = Vec3(centre.x, centre.y + radius, centre.z)
      val (x, y) = inFrameWith(halfTan)(scene.camera, top)
      withClue(s"frame $i: "):
        math.abs(x) should be <= 0.85f + framingTolerance
        math.abs(y) should be <= 0.85f + framingTolerance

  it should "move the camera back for a narrower field and a larger margin" in:
    val t = 51.2f
    def away(framing: Framing) =
      val c = DimensionLadder.sceneAt(t, look = Look.default.copy(framing = framing)).camera
      (c.position - c.lookAt).magnitude
    away(Framing(fovDegrees = 35f)) should be > away(Framing.default)
    away(Framing(margin = 0.85f)) should be > away(Framing.default)

  it should "reject a field of view or a margin that cannot frame anything" in:
    List(0f, -1f, 180f, 200f).foreach: fov =>
      an[IllegalArgumentException] should be thrownBy Framing(fovDegrees = fov)
    List(0f, -0.5f, 1.5f).foreach: margin =>
      an[IllegalArgumentException] should be thrownBy Framing(margin = margin)

  private def rotationOfCube(look: Look, t: Float = 0f): Vec3 =
    objectOf(DimensionLadder.sceneAt(t, look = look)) match
      case cube: menger.dsl.Cube => cube.rotation
      case sponge: menger.dsl.Sponge => sponge.rotation
      case other => fail(s"expected a cube or a sponge, got $other")

  private def turnedBy(planes: (String, Float)*): Look =
    Look.default.copy(turns = _ => _ => planes.toList.map((plane, degrees) =>
      menger.spec.PlaneRotation(plane, degrees)))

  private def foldTo(n: Int): Segment =
    DimensionLadder.segments(cycle).find(s => s.kind == Kind.Fold && s.dimension == n).get

  "Look.turns" should "replace the turn schedule: none at all stops the turn" in:
    rotationOfCube(Look.default, 3f).y should be > 0f
    rotationOfCube(turnedBy(), 3f) shouldBe Vec3(0f, 0f, 0f)

  it should "turn a 3D object in xz about y, as the turntable always did" in:
    rotationOfCube(turnedBy("xz" -> 90f)) shouldBe Vec3(0f, math.toRadians(90.0).toFloat, 0f)

  it should "turn a 3D object in yz about x and in xy about z, second axis to the first" in:
    val half = math.toRadians(30.0).toFloat
    rotationOfCube(turnedBy("yz" -> 30f)) shouldBe Vec3(-half, 0f, 0f)
    rotationOfCube(turnedBy("xy" -> 30f)) shouldBe Vec3(0f, 0f, -half)
    rotationOfCube(turnedBy("yz" -> 20f, "xz" -> 45f, "xy" -> 10f)) shouldBe
      Vec3(-math.toRadians(20.0).toFloat, math.toRadians(45.0).toFloat,
        -math.toRadians(10.0).toFloat)

  it should "turn a 3D sponge in every plane too" in:
    val look = turnedBy("yz" -> 30f, "xz" -> 45f)
    val breath = DimensionLadder.segments(cycle).find(s => s.kind == Kind.Breath).get
    DimensionLadder.objectAt(breath, 0.25f, look = look) match
      case sponge: menger.dsl.Sponge =>
        sponge.rotation shouldBe
          Vec3(-math.toRadians(30.0).toFloat, math.toRadians(45.0).toFloat, 0f)
      case other => fail(s"expected a sponge, got $other")

  it should "hand the planes of the schedule to a higher object's projection" in:
    val look = turnedBy("xw" -> 40f, "yz" -> 15f)
    val fold = foldTo(4)
    val spec = DimensionLadder.objectAt(fold, 0.5f, look = look).toObjectSpec
    spec.projection match
      case Some(view: menger.spec.ProjectionSpec) =>
        view.planes.map(p => (p.plane.letters, p.angle.value)) shouldBe
          List(("xw", 40f), ("yz", 15f))
      case other => fail(s"expected a ProjectionSpec, got $other")

  it should "hand a 3D object over to the 4D one with the same orientation, planes in order" in:
    // Late enough that the 4D object is drawn, not the 3D stand-in (standInExtrusion), and
    // early enough that its new axis is still nearly flat.
    val firstDrawn4D = 0.02f
    for planes <- List(List("yz" -> 30f, "xz" -> 45f), List("xz" -> 45f), List("yz" -> 30f),
        List("yz" -> 30f, "xz" -> 45f, "xy" -> 20f))
    do
      val look = turnedBy(planes*)
      withClue(s"planes $planes: "):
        val after = DimensionLadder.objectAt(foldTo(4), firstDrawn4D, look = look)
        after.toObjectSpec.objectType shouldBe "n-cube"
        sameShape(DimensionLadder.objectAt(foldTo(4), 0f, look = look), after)

  it should "reject the planes of a 3D object in another order than yz, xz, xy" in:
    for planes <- List(List("xz" -> 45f, "yz" -> 30f), List("xy" -> 20f, "xz" -> 45f),
        List("xy" -> 20f, "yz" -> 30f))
    do
      val look = turnedBy(planes*)
      withClue(s"planes $planes: "):
        val e = the[IllegalArgumentException] thrownBy DimensionLadder.sceneAt(0f, look = look)
        e.getMessage should include("yz, xz, xy")

  it should "leave the order of the other planes to a higher object, as listed" in:
    val look = turnedBy("xz" -> 45f, "xw" -> 40f, "yz" -> 30f)
    val spec = DimensionLadder.objectAt(foldTo(4), 0.5f, look = look).toObjectSpec
    spec.projection match
      case Some(view: menger.spec.ProjectionSpec) =>
        view.planes.map(_.plane.letters) shouldBe List("xz", "xw", "yz")
      case other => fail(s"expected a ProjectionSpec, got $other")

  it should "change nothing but the turn" in:
    for t <- List(3f, 22.7f, 102f) do
      val plain = DimensionLadder.sceneAt(t)
      val changed = DimensionLadder.sceneAt(t, look = turnedBy())
      changed.camera shouldBe plain.camera
      changed.objects.head shouldBe plain.objects.head
      changed.lights shouldBe plain.lights
      val (after, before) = (objectOf(changed).toObjectSpec, objectOf(plain).toObjectSpec)
      after.objectType shouldBe before.objectType
      after.level shouldBe before.level
      after.size shouldBe before.size
      after.material shouldBe before.material

  "Look.restDepth" should "default to the stored depths and, when set, place the centre by it" in:
    Look.default.restDepth shouldBe None
    val depth = Stage.restDepth(DimensionLadder.topDimension) + 0.5f
    val deeper = Look.default.copy(restDepth = Some(depth))
    val plain = DimensionLadder.sceneAt(0f)
    val changed = DimensionLadder.sceneAt(0f, look = deeper)
    objectOf(changed).toObjectSpec.y shouldBe objectOf(plain).toObjectSpec.y + 0.5f +- 1e-5f
    changed.camera.lookAt.y should be > plain.camera.lookAt.y
