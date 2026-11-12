package menger.engines.scene

import menger.cli.ObjectSpecParser
import menger.common.Material
import menger.common.ProfilingConfig
import menger.engines.GeometryRegistry
import menger.engines.SceneGroups
import menger.spec.ObjectSpec
import menger.spec.PlaneName
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The n-cube through the engines (menger#163): CPU faces through the chain, one tube per edge,
  * the right builder, and no 4D fast path. */
class NCubeSceneSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private def turn(plane: String, degrees: Float) = PlaneRotation(PlaneName(plane), degrees)

  private def hypercube(
    dimension: Int,
    projection: Option[ProjectionSpec] = None,
    edges: Boolean = false
  ): ObjectSpec =
    ObjectSpec(
      "n-cube", size = 1.2f, dimension = Some(dimension), projection = projection,
      edgeRadius = Option.when(edges)(0.02f), edgeMaterial = Option.when(edges)(Material.Film)
    )

  private def scaled(spec: ObjectSpec, lastFactor: Float): ObjectSpec =
    val dimension = spec.dimension.get
    spec.copy(scale = Some(IndexedSeq.fill(dimension - 1)(1f) :+ lastFactor))

  private val RoundingFactor = 10000f
  private val MinTriangleArea = 1e-6f

  /** The corners of the faces that are not degenerate, rounded to 1e-4: a collapsed axis turns
    * the faces that use it into lines, which draw nothing. */
  private def nonDegenerateVertices(spec: ObjectSpec): Set[Seq[Int]] =
    val mesh = MeshFactory.create(spec)
    val points = mesh.vertices.grouped(mesh.vertexStride).map(_.take(3).toIndexedSeq).toIndexedSeq
    def area(a: IndexedSeq[Float], b: IndexedSeq[Float], c: IndexedSeq[Float]): Float =
      val u = a.zip(b).map((p, q) => q - p)
      val v = a.zip(c).map((p, q) => q - p)
      val cross = Seq(
        u(1) * v(2) - u(2) * v(1), u(2) * v(0) - u(0) * v(2), u(0) * v(1) - u(1) * v(0)
      )
      math.sqrt(cross.map(x => x * x).sum).toFloat / 2
    mesh.indices.grouped(3)
      .filter(t => area(points(t(0)), points(t(1)), points(t(2))) > MinTriangleArea)
      .flatMap(_.map(points)).map(_.map(x => math.round(x * RoundingFactor))).toSet

  private def positions(spec: ObjectSpec): Set[Seq[Float]] =
    val mesh = MeshFactory.create(spec)
    mesh.vertices.grouped(mesh.vertexStride).map(_.take(3).toSeq).toSet

  "MeshFactory.create" should "make two triangles of each of the 80 faces of a penteract" in:
    val mesh = MeshFactory.create(hypercube(5))
    mesh.numTriangles shouldBe 160
    mesh.vertices.forall(_.isFinite) shouldBe true

  it should "make 1792 faces of an 8-cube" in:
    MeshFactory.create(hypercube(8)).numTriangles shouldBe 2 * 1792

  it should "use ProjectionSpec(N) when the spec has no projection" in:
    val implicitView = hypercube(6)
    val explicitView = hypercube(6, Some(ProjectionSpec(6)))
    MeshFactory.create(implicitView).vertices shouldBe MeshFactory.create(explicitView).vertices

  it should "show the 4-cube with the vertices of the Tesseract through the same view" in:
    val view = ProjectionSpec(4)
    val tesseract = ObjectSpec("tesseract", size = 1.2f, projection = Some(view))
    positions(hypercube(4, Some(view))) shouldBe positions(tesseract)

  it should "turn in planes using the axes of a higher view dimension" in:
    val flat = hypercube(5, Some(ProjectionSpec(7)))
    val turned = hypercube(
      5, Some(ProjectionSpec(7, planes = List(turn("vt", 40f), turn("tu", 30f), turn("xt", 25f))))
    )
    MeshFactory.create(turned).vertices should not be MeshFactory.create(flat).vertices

  it should "act on the cube only through planes that reach it" in:
    // A 5-cube has nothing in t or s: the plane ts alone turns zeros, a plane with an axis it
    // has (x..v) must move it there first.
    val flat = hypercube(5, Some(ProjectionSpec(8)))
    val onlyBeyond = hypercube(5, Some(ProjectionSpec(8, planes = List(turn("ts", 40f)))))
    val expected = MeshFactory.create(flat).vertices
    MeshFactory.create(onlyBeyond).vertices.zip(expected).foreach { (actual, wanted) =>
      actual shouldBe wanted +- 1e-6f
    }

  it should "drop faces and edges past a stage's eye and keep the rest" in:
    val close = ProjectionSpec(
      5, stages = List(ProjectionStage(0.6f, 0.3f), ProjectionStage(6f, 1.5f))
    )
    val spec = hypercube(5, Some(close)).copy(size = 2f)
    MeshFactory.create(spec).numTriangles shouldBe 2 * 24
    NCubeScene.edgeEndpoints(spec).flatten should have size 32
    NCubeScene.edgeEndpoints(spec) should have size 80

  // menger#164: the per-axis scale extrudes along the newest axis.
  it should "scale the last axis of a penteract and leave the others" in:
    val size = 1.2f
    val spec = hypercube(5).copy(size = size, scale = Some(IndexedSeq(1f, 1f, 1f, 1f, 0.3f)))
    val extents = (0 until 5).map { axis =>
      val coordinates = NCubeScene.faces(spec).sourceMesh.vertices.map(_(axis))
      coordinates.max - coordinates.min
    }
    extents.take(4).foreach(_ shouldBe size +- 1e-5f)
    extents(4) shouldBe 0.3f * size +- 1e-5f

  it should "show Hypercube(n) with the last scale 0 as Hypercube(n-1), for n = 5..8" in:
    (5 to 8).foreach { n =>
      val view = Some(ProjectionSpec(n))
      nonDegenerateVertices(scaled(hypercube(n, view), 0f)) shouldBe
        nonDegenerateVertices(hypercube(n - 1, view))
    }

  it should "show Cube(s) as Hypercube(4) with w scale 0 and size s * eye / (eye - screen)" in:
    val cubeSize = 1.2f
    val view = ProjectionSpec(4)
    val flat = scaled(hypercube(4, Some(view)), 0f)
      .copy(size = cubeSize * view.eye / (view.eye - view.screen))
    nonDegenerateVertices(flat) shouldBe nonDegenerateVertices(ObjectSpec("cube", size = cubeSize))

  it should "leave out the edges that a collapsed axis left without length" in:
    val flat = scaled(hypercube(5, edges = true), 0f)
    NCubeScene.edgeEndpoints(flat) should have size 80
    NCubeScene.edgeEndpoints(flat).flatten should have size 64
    NCubeScene.edgeEndpoints(hypercube(5, edges = true)).flatten should have size 80

  "MeshFactory.createUpload" should "build a CPU mesh for an n-cube" in:
    MeshFactory.createUpload(hypercube(5)) shouldBe a[MeshUploadPlan.Cpu]

  "NCubeScene" should "reject a view with fewer dimensions than the cube, naming both" in:
    val refusal = the[IllegalArgumentException] thrownBy
      NCubeScene.viewOf(hypercube(6, None).copy(projection = Some(ProjectionSpec(5))))
    refusal.getMessage should (include("6-cube") and include("5"))

  it should "need a dimension" in:
    an[menger.common.ConfigurationException] should be thrownBy
      NCubeScene.dimensionOf(ObjectSpec("n-cube"))

  it should "place the edge endpoints at the object's position" in:
    val spec = hypercube(5).copy(x = 1f, y = 2f, z = 3f)
    val placed = NCubeScene.edgeEndpoints(spec)
    val atOrigin = NCubeScene.edgeEndpoints(spec.copy(x = 0f, y = 0f, z = 0f))
    placed should have size atOrigin.size.toLong
    val offset = Seq(1f, 2f, 3f)
    def shift(point: menger.common.Vector[3], origin: menger.common.Vector[3]) =
      Seq(point(0) - origin(0), point(1) - origin(1), point(2) - origin(2))
    placed.zip(atOrigin).foreach { (actual, expected) =>
      (actual, expected) match
        case (Some((a0, a1)), Some((e0, e1))) =>
          Seq(shift(a0, e0), shift(a1, e1)).foreach { shifted =>
            shifted.zip(offset).foreach((moved, wanted) => moved shouldBe wanted +- 1e-5f)
          }
        case _ => fail("no edge is clipped at the default view")
    }

  "The builders" should "take an edge-rendered n-cube with TesseractEdgeSceneBuilder" in:
    GeometryRegistry.builderFor(List(hypercube(5, edges = true))).get shouldBe
      a[TesseractEdgeSceneBuilder]

  it should "take an n-cube without edges with the triangle mesh builder" in:
    GeometryRegistry.builderFor(List(hypercube(5))).get shouldBe a[TriangleMeshSceneBuilder]

  it should "build an edge-rendered n-cube together with edge-rendered tesseracts" in:
    val tesseract = ObjectSpec("tesseract", edgeRadius = Some(0.02f))
    GeometryRegistry.builderFor(List(hypercube(5, edges = true), tesseract)).get shouldBe
      a[TesseractEdgeSceneBuilder]

  it should "order an edge-rendered n-cube into the edge group, first" in:
    val sphere = ObjectSpec("sphere")
    val groups = SceneGroups.buildOrder(List(sphere, hypercube(5, edges = true), hypercube(4)))
    groups.head.map(_.objectType) shouldBe List("n-cube")
    groups.head.head.hasEdgeRendering shouldBe true

  "TesseractEdgeSceneBuilder" should "need an instance for the faces and one per edge" in:
    val builder = TesseractEdgeSceneBuilder(".")
    builder.calculateRequiredInstances(List(hypercube(5, edges = true))) shouldBe 1 + 80
    builder.calculateRequiredInstances(List(hypercube(8, edges = true))) shouldBe 1 + 1024

  it should "validate an 8-cube against the instance limit" in:
    val builder = TesseractEdgeSceneBuilder(".")
    val eightCube = List(hypercube(8, edges = true))
    builder.validate(eightCube, 2048) shouldBe Right(())
    builder.validate(eightCube, 64).left.toOption.get should include("1025")

  it should "count the edges of the instance budget without building a mesh" in:
    TesseractEdgeSceneBuilder(".").calculateInstanceCount(List(hypercube(6, edges = true))) shouldBe
      1L + 6 * 32

  "TrackedEdges4D" should "not take an n-cube scene: it always rebuilds" in:
    TrackedEdges4D.isEdgeScene(List(hypercube(5, edges = true))) shouldBe false

  "SceneCost.triangles" should "count the two triangles of each face" in:
    SceneCost.triangles(hypercube(5)) shouldBe 160L
    SceneCost.triangles(hypercube(8)) shouldBe 3584L

  "The command line" should "refuse an n-cube and point to the DSL" in:
    val refusal = ObjectSpecParser.parse("type=n-cube")
    refusal.isLeft shouldBe true
    refusal.left.toOption.get should (include("DSL") and include("tesseract"))
