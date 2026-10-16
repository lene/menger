package menger.engines.scene

import scala.collection.mutable

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
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
class EdgeRotationFastPathSuite extends AnyFlatSpec with Matchers with BeforeAndAfterEach:
  given ProfilingConfig = ProfilingConfig.disabled

  private val Size = ImageSize(200, 150)
  private val MaxInstances = 4096

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
    ObjectSpec.parse(s).fold(e => fail(s"bad spec '$s': $e"), identity)

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

  it should "decline when the rotation changes which edges the eye_w plane clips" in:
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
