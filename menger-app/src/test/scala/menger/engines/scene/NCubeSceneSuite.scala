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
    val moved = NCubeScene.edgeEndpoints(spec)
    val atOrigin = NCubeScene.edgeEndpoints(spec.copy(x = 0f, y = 0f, z = 0f))
    moved.zip(atOrigin).foreach { case (Some((m0, _)), Some((o0, _))) =>
      Seq(m0(0) - o0(0), m0(1) - o0(1), m0(2) - o0(2)).zip(Seq(1f, 2f, 3f)).foreach {
        (moved, offset) => moved shouldBe offset +- 1e-5f
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
