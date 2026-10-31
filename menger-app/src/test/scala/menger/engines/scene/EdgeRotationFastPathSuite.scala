package menger.engines.scene

import scala.collection.mutable

import io.github.lene.optix.OptiXRenderer
import io.github.lene.qa.Perf
import io.github.lene.qa.PerfGate
import io.github.lene.qa.RelativeBenchmark
import io.github.lene.qa.Side
import menger.cli.ObjectSpecParser
import menger.spec.ObjectSpec
import menger.common.ImageSize
import menger.common.ProfilingConfig
import menger.common.Vector
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Interactive 4D rotation of edge-rendered objects moves the built geometry in place
  * (TesseractEdgeSceneBuilder.updateProjection) instead of rebuilding the scene (usability
  * review 2026-09, F22). The moved scene has to render exactly like a fresh build. */
class EdgeRotationFastPathSuite extends AnyFlatSpec with Matchers with BeforeAndAfterEach
    with PerfGate:
  given ProfilingConfig = ProfilingConfig.disabled

  private val Size = ImageSize(200, 150)
  private val MaxInstances = 4096
  // The claim is "faster than" (ratio < 1), as Project4DGpuSuite's update-vs-rebuild gate.
  private val MaxRatioMovedVsRebuilt = 1.0

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[OptiXRenderer] = None

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def renderer: OptiXRenderer =
    rendererOpt.getOrElse(throw new IllegalStateException("Renderer not initialized"))

  override def withFixture(test: NoArgTest): Outcome =
    if rendererOpt.isDefined then super.withFixture(test)
    else cancel("OptiX native library not available")

  override def beforeEach(): Unit =
    super.beforeEach()
    try
      val _ = OptiXRenderer.isLibraryLoaded
      val r = new OptiXRenderer()
      if r.initialize(MaxInstances) then
        r.setCamera(Vector[3](0f, 0.5f, 4f), Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f), 45f)
        r.setLight(Vector[3](0.4f, -0.6f, -0.5f), 1f)
        rendererOpt = Some(r)
    catch case _: Throwable => ()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally
      rendererOpt = None
      super.afterEach()

  private def spec(s: String): ObjectSpec =
    ObjectSpecParser.parse(s).fold(e => fail(s"bad spec '$s': $e"), identity)

  private def polytope(rotXW: Int, extra: String = ""): ObjectSpec =
    spec(s"type=24-cell:size=1:material=film:edge-material=gold:edge-radius=0.02:rot-xw=$rotXW$extra")

  private def buildTracked(specs: List[ObjectSpec]): IndexedSeq[TesseractEdgeSceneBuilder.EdgeTrack] =
    val tracks = mutable.Map.empty[Int, TesseractEdgeSceneBuilder.EdgeTrack]
    TesseractEdgeSceneBuilder(".", (i, t) => tracks(i) = t)
      .validateAndBuild(specs, renderer, MaxInstances).get
    specs.indices.map(tracks)

  private def render(): Array[Byte] = renderer.render(Size)

  /** Renders `specs` on a cleared renderer: matte, no shadows, so objects far apart can't
    * affect each other's pixels. */
  private def renderAlone(specs: List[ObjectSpec]): Array[Byte] =
    renderer.clearAllInstances()
    renderer.setShadows(false)
    if specs.nonEmpty then buildTracked(specs)
    render()

  private def matte(rotXW: Int, x: Float): ObjectSpec =
    spec(s"type=24-cell:size=1:pos=$x,0,0:material=matte:edge-material=matte:" +
      s"edge-radius=0.02:rot-xw=$rotXW")

  private def pixels(image: Array[Byte]): IndexedSeq[Seq[Int]] =
    image.grouped(4).map(_.toSeq.map(_ & 0xff)).toIndexedSeq

  // Usability review 2026-09, session 2 (F44, menger#52): every 4D object already has its own
  // mesh and edges, built from its own projection; a scene check still demanded one shared
  // projection and rejected "a 24-cell above the tesseract sponge".
  "TesseractEdgeSceneBuilder" should "render 4D objects with different projections each as if alone" in:
    val left = matte(30, -1.4f)
    val right = matte(0, 1.4f)
    // An empty scene renders a different background, so park one object far out of view.
    val background = pixels(renderAlone(List(matte(0, 100f))))
    val leftAlone = pixels(renderAlone(List(left)))
    val rightAlone = pixels(renderAlone(List(right)))
    val both = pixels(renderAlone(List(left, right)))

    def coveredOnlyBy(own: IndexedSeq[Seq[Int]], other: IndexedSeq[Seq[Int]]): IndexedSeq[Int] =
      own.indices.filter(i => own(i) != background(i) && other(i) == background(i))
    val leftPixels = coveredOnlyBy(leftAlone, rightAlone)
    val rightPixels = coveredOnlyBy(rightAlone, leftAlone)
    withClue(s"non-background: left ${leftAlone.indices.count(i => leftAlone(i) != background(i))}, " +
      s"right ${rightAlone.indices.count(i => rightAlone(i) != background(i))}, " +
      s"both ${both.indices.count(i => both(i) != background(i))}: ") {
      leftPixels.size should be > 500
      rightPixels.size should be > 500
    }

    def mismatches(expected: IndexedSeq[Seq[Int]], at: IndexedSeq[Int]): Int =
      at.count(i => expected(i).lazyZip(both(i)).exists((a, b) => math.abs(a - b) > 8))
    mismatches(leftAlone, leftPixels) should be <= leftPixels.size / 100
    mismatches(rightAlone, rightPixels) should be <= rightPixels.size / 100

    // The test is only meaningful if the projections really render differently.
    pixels(renderAlone(List(matte(0, -1.4f)))) should not equal leftAlone

  "TesseractEdgeSceneBuilder.updateProjection" should "render exactly like a fresh build" in:
    val before = List(polytope(30))
    val after = List(polytope(40))
    val tracks = buildTracked(before)
    val first = render()  // builds the IAS, so the update has to refit it

    TesseractEdgeSceneBuilder.updateProjection(before, after, tracks, renderer) shouldBe true
    val moved = render()

    renderer.clearAllInstances()
    buildTracked(after)
    val fresh = render()

    moved should not equal first
    moved shouldEqual fresh

  // Session 5, F112 (menger#129): the animated window rebuilt every cylinder each frame.
  "TrackedEdges4D" should "move an animated window's edge scene in place, as a fresh build renders it" in:
    val before = List(polytope(30))
    val after = List(polytope(40))
    val state = TrackedEdges4D.build(before, renderer, ".", _ => MaxInstances).get
      .getOrElse(fail("no tracks recorded"))
    val first = render()

    val moved = TrackedEdges4D.update(state, after, renderer)
    moved.map(_.specs) shouldBe Some(after)
    val movedImage = render()

    renderer.clearAllInstances()
    buildTracked(after)
    movedImage shouldEqual render()
    movedImage should not equal first
    // Anything but the projection: nothing moved, the caller rebuilds.
    TrackedEdges4D.update(moved.get, List(polytope(40).copy(x = 0.5f)), renderer) shouldBe None

  // Session 6 (menger#129): an animation's hold phase rebuilt every unchanged frame, 1.1 s each.
  it should "take an unchanged frame without a rebuild" in:
    val state = TrackedEdges4D.build(List(polytope(30)), renderer, ".", _ => MaxInstances).get
      .getOrElse(fail("no tracks recorded"))

    TrackedEdges4D.update(state, List(polytope(30)), renderer).map(_.specs) shouldBe
      Some(List(polytope(30)))

  // Session 6 (menger#129): an edged sponge next to an unedged 24-cell had no in-place path and
  // rebuilt everything every frame, 1.12 s each.
  private def plainPolytope(rotXW: Int): ObjectSpec =
    spec(s"type=24-cell:size=0.6:pos=1.2,0,0:material=film:rot-xw=$rotXW")

  "TrackedMixed4D" should "move a partly edged scene in place, as a fresh build renders it" in:
    val scene = (r: Int) => List(polytope(r), plainPolytope(r))
    val state = TrackedMixed4D.build(scene(30), renderer, ".", MaxInstances).get
      .getOrElse(fail("no tracks recorded"))
    val first = render()

    val moved = TrackedMixed4D.update(state, scene(40), renderer)
    moved.map(s => s.edges.specs ++ s.meshes.specs) shouldBe Some(scene(40))
    val movedImage = render()

    renderer.clearAllInstances()
    TrackedMixed4D.build(scene(40), renderer, ".", MaxInstances).get
    movedImage shouldEqual render()
    movedImage should not equal first
    TrackedMixed4D.update(moved.get, scene(40), renderer) should not be None
    // An edged object moved: nothing moved, the caller rebuilds.
    val shifted = List(polytope(40).copy(x = 0.5f), plainPolytope(40))
    TrackedMixed4D.update(moved.get, shifted, renderer) shouldBe None

  it should "build like the group-by-group build of a mixed scene" in:
    val scene = List(polytope(30), plainPolytope(30))
    renderer.clearAllInstances()
    menger.engines.SceneGroups.buildOrder(scene).foreach { group =>
      menger.engines.GeometryRegistry.builderFor(group, ".").getOrElse(fail(s"no builder: $group"))
        .validateAndBuild(group, renderer, MaxInstances).get
    }
    val grouped = render()

    renderer.clearAllInstances()
    TrackedMixed4D.build(scene, renderer, ".", MaxInstances).get
    render() shouldEqual grouped

  it should "take only mixed scenes of projected 4D meshes" in:
    TrackedMixed4D.isMixedScene(List(polytope(0), plainPolytope(0))) shouldBe true
    TrackedMixed4D.isMixedScene(List(polytope(0), polytope(10))) shouldBe false
    TrackedMixed4D.isMixedScene(List(polytope(0), spec("type=sphere:size=1"))) shouldBe false
    TrackedMixed4D.isMixedScene(Nil) shouldBe false

  // The edge scene's animation frame, moved in place vs rebuilt, both with their render. Measured
  // on session 5's Task 4.7 scene (27,648 cylinders) 2026-10-23: 3.0 s rebuilt, 0.74 s moved.
  "TrackedEdges4D" should "animate an edge scene's 4D rotation faster in place than rebuilt" taggedAs Perf in:
    val sponge = (rotXW: Int) => spec(
      s"type=tesseract-sponge:level=1:size=1:material=film:edge-material=gold:" +
        s"edge-radius=0.01:rot-xw=$rotXW")
    val frame = new java.util.concurrent.atomic.AtomicInteger(0)
    def nextSpecs(): List[ObjectSpec] = List(sponge(frame.incrementAndGet() % 36 * 10 + 5))
    val state = new java.util.concurrent.atomic.AtomicReference[Option[TrackedEdges4D.State]](None)
    def build(specs: List[ObjectSpec]): Unit =
      renderer.clearAllInstances()
      state.set(TrackedEdges4D.build(specs, renderer, ".", _ => MaxInstances).get)
    def resetScene(): Unit =
      build(List(sponge(0)))
      val _ = render()
    val moved = Side(
      "TrackedEdges4D.update",
      () => {
        state.set(state.get.flatMap(TrackedEdges4D.update(_, nextSpecs(), renderer)))
        val _ = render()
      },
      prepare = () => resetScene()
    )
    val rebuilt = Side(
      "rebuild",
      () => { build(nextSpecs()); val _ = render() },
      prepare = () => resetScene()
    )
    assertWithin("tesseract sponge L1 with edge tubes, moved vs rebuilt",
      RelativeBenchmark.compare(rebuilt, moved, MaxRatioMovedVsRebuilt))

  it should "take exactly the scenes the edge builder takes" in:
    TrackedEdges4D.isEdgeScene(List(polytope(0), polytope(10))) shouldBe true
    TrackedEdges4D.isEdgeScene(List(polytope(0), spec("type=24-cell:size=1"))) shouldBe false
    TrackedEdges4D.isEdgeScene(List(spec("type=cube:size=1:edge-radius=0.02"))) shouldBe false
    TrackedEdges4D.isEdgeScene(Nil) shouldBe false

  // The menger#129 gate for the mixed path: session 6's Task 4.7 scene (an edged tesseract sponge
  // and a plain 24-cell) rebuilt every frame, 1.12 s; moved in place it ran at ~6 fps.
  "TrackedMixed4D" should "animate a partly edged scene's 4D rotation faster in place than rebuilt" taggedAs
    Perf in:
    val scene = (rotXW: Int) => List(
      spec(s"type=tesseract-sponge:level=1:size=1:material=film:edge-material=gold:" +
        s"edge-radius=0.01:rot-xw=$rotXW"),
      plainPolytope(rotXW))
    val frame = new java.util.concurrent.atomic.AtomicInteger(0)
    def nextSpecs(): List[ObjectSpec] = scene(frame.incrementAndGet() % 36 * 10 + 5)
    val state = new java.util.concurrent.atomic.AtomicReference[Option[TrackedMixed4D.State]](None)
    def build(specs: List[ObjectSpec]): Unit =
      renderer.clearAllInstances()
      state.set(TrackedMixed4D.build(specs, renderer, ".", MaxInstances).get)
    def resetScene(): Unit =
      build(scene(0))
      val _ = render()
    val moved = Side(
      "TrackedMixed4D.update",
      () => {
        state.set(state.get.flatMap(TrackedMixed4D.update(_, nextSpecs(), renderer)))
        val _ = render()
      },
      prepare = () => resetScene()
    )
    val rebuilt = Side(
      "rebuild",
      () => { build(nextSpecs()); val _ = render() },
      prepare = () => resetScene()
    )
    assertWithin("edged tesseract sponge L1 and plain 24-cell, moved vs rebuilt",
      RelativeBenchmark.compare(rebuilt, moved, MaxRatioMovedVsRebuilt))

  "TesseractEdgeSceneBuilder.updateProjection" should "decline when the rotation changes which edges the eye_w plane clips" in:
    // Inside the vertices' 4D radius (0.5 at size 1), so some rotations clip some edges.
    val eyeW = ":eye-w=0.4:screen-w=0.2"
    val base = polytope(0, eyeW)
    val baseMask = TesseractEdgeSceneBuilder.edgeEndpoints(base, buildTracked(List(base)).head.edges)
      .map(_.isDefined)
    val edges = buildTracked(List(base)).head.edges
    val rotated = (5 to 175 by 5).map(r => polytope(r, eyeW))
      .find(s => TesseractEdgeSceneBuilder.edgeEndpoints(s, edges).map(_.isDefined) != baseMask)
      .getOrElse(fail("no rotation changes the clip set; pick a smaller eye-w"))
    renderer.clearAllInstances()
    val tracks = buildTracked(List(base))

    TesseractEdgeSceneBuilder.updateProjection(List(base), List(rotated), tracks, renderer) shouldBe false

  it should "decline for a fractional-level sponge, whose faces are CPU-projected" in:
    val sponge = (r: Int) =>
      spec(s"type=tesseract-sponge:level=0.5:size=1:material=film:edge-material=gold:rot-xw=$r")
    val tracks = buildTracked(List(sponge(10)))
    tracks.head.faces shouldBe TesseractEdgeSceneBuilder.FaceTrack.Cpu

    TesseractEdgeSceneBuilder.updateProjection(List(sponge(10)), List(sponge(20)), tracks, renderer) shouldBe false

  it should "decline when anything but the projection changed" in:
    val tracks = buildTracked(List(polytope(30)))
    val moved = polytope(30).copy(x = 0.5f)

    TesseractEdgeSceneBuilder.updateProjection(List(polytope(30)), List(moved), tracks, renderer) shouldBe false
