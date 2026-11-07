package menger.engines.scene

import menger.common.ProfilingConfig
import menger.common.x
import menger.common.y
import menger.common.z
import menger.objects.higher_d.Dimension
import menger.spec.ObjectSpec
import menger.spec.PlaneName
import menger.spec.PlaneRotation
import menger.spec.Projection4DSpec
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** An object with a `ProjectionSpec` on the CPU path (menger#162): same image as the 4D view in
  * 4D, a different one when lifted, and never a silent fall back to a 4D view. */
class NdProjectionSceneSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private val Tolerance = 1e-5f

  private def turn(plane: String, degrees: Float) = PlaneRotation(PlaneName(plane), degrees)

  private def tesseract(projection: Option[menger.spec.ProjectionLike]): ObjectSpec =
    ObjectSpec("tesseract", size = 1.3f, projection = projection)

  private def vertices(spec: ObjectSpec): Array[Float] = MeshFactory.create(spec).vertices

  private def maxDifference(a: Array[Float], b: Array[Float]): Float =
    a.length shouldBe b.length
    a.zip(b).map((x, y) => math.abs(x - y)).maxOption.getOrElse(0f)

  "The letter tables" should "agree between the scene spec and the objects" in:
    PlaneName.axisLetters shouldBe Dimension.axisNames

  "A 4D ProjectionSpec" should "render the legacy view of a single turn" in:
    for (letters, legacy) <- Seq(
        ("xw", Projection4DSpec(rotXW = 33f, rotYW = 0f, rotZW = 0f)),
        ("yw", Projection4DSpec(rotXW = 0f, rotYW = 33f, rotZW = 0f)),
        ("zw", Projection4DSpec(rotXW = 0f, rotYW = 0f, rotZW = 33f))
      )
    do
      val nd = ProjectionSpec(4, planes = List(turn(letters, 33f)))
      maxDifference(vertices(tesseract(Some(nd))), vertices(tesseract(Some(legacy)))) should
        be < Tolerance

  it should "render the legacy view of XW, YW and ZW together when written zw, yw, xw" in:
    // The legacy composition turns ZW first: the first plane written is the first turned.
    val legacy = Projection4DSpec(
      eyeW = 3.5f, screenW = 1.2f, rotXW = 25f, rotYW = -40f, rotZW = 70f
    )
    val nd = ProjectionSpec(
      4, planes = List(turn("zw", 70f), turn("yw", -40f), turn("xw", 25f)),
      eye = 3.5f, screen = 1.2f
    )
    maxDifference(vertices(tesseract(Some(nd))), vertices(tesseract(Some(legacy)))) should
      be < Tolerance

  it should "render a polytope and a sponge like the legacy view" in:
    val legacy = Projection4DSpec(rotXW = 20f, rotYW = 0f, rotZW = 0f)
    val nd = ProjectionSpec(4, planes = List(turn("xw", 20f)))
    for spec <- Seq(
        ObjectSpec("16-cell", size = 1.1f), ObjectSpec("tesseract-sponge", level = Some(1f))
      )
    do
      maxDifference(
        vertices(spec.copy(projection = Some(nd))), vertices(spec.copy(projection = Some(legacy)))
      ) should be < Tolerance

  "A 5D ProjectionSpec" should "differ from the 4D view of the same turn" in:
    val fourD = tesseract(Some(Projection4DSpec(rotXW = 0f, rotYW = 0f, rotZW = 0f)))
    val lifted = tesseract(Some(ProjectionSpec(5, planes = List(turn("xv", 30f)))))
    maxDifference(vertices(lifted), vertices(fourD)) should be > 0.01f

  it should "differ from the unturned 5D lift" in:
    val unturned = tesseract(Some(ProjectionSpec(5)))
    val turned = tesseract(Some(ProjectionSpec(5, planes = List(turn("xv", 30f)))))
    maxDifference(vertices(turned), vertices(unturned)) should be > 0.01f

  it should "turn in the order written" in:
    val xvThenWv = ProjectionSpec(5, planes = List(turn("xv", 30f), turn("wv", 40f)))
    val wvThenXv = ProjectionSpec(5, planes = List(turn("wv", 40f), turn("xv", 30f)))
    maxDifference(vertices(tesseract(Some(xvThenWv))), vertices(tesseract(Some(wvThenXv)))) should
      be > 0.001f

  it should "honour the per-stage eye and screen" in:
    val uniform = ProjectionSpec(5, planes = List(turn("xv", 30f)))
    val perStage = uniform.copy(stages = List(ProjectionStage(6f, 1f), ProjectionStage(3f, 1.5f)))
    maxDifference(vertices(tesseract(Some(uniform))), vertices(tesseract(Some(perStage)))) should
      be > 0.01f

  it should "give the same mesh as the same view with the stages written out" in:
    val implicitStages = ProjectionSpec(6, planes = List(turn("xv", 30f)), eye = 4f, screen = 2f)
    val written = implicitStages.copy(stages = List.fill(3)(ProjectionStage(4f, 2f)))
    vertices(tesseract(Some(implicitStages))) shouldBe vertices(tesseract(Some(written)))

  it should "change when only a plane angle changes, with no cache serving the old mesh" in:
    val first = vertices(tesseract(Some(ProjectionSpec(5, planes = List(turn("xv", 10f))))))
    val second = vertices(tesseract(Some(ProjectionSpec(5, planes = List(turn("xv", 50f))))))
    maxDifference(first, second) should be > 0.001f

  "MeshFactory.createUpload" should "build a CPU mesh, not a GPU 4D plan, for a ProjectionSpec" in:
    val plan = MeshFactory.createUpload(tesseract(Some(ProjectionSpec(5))))
    plan shouldBe a[MeshUploadPlan.Cpu]

  it should "keep the GPU plan for the 4D view" in:
    MeshFactory.createUpload(tesseract(Some(Projection4DSpec()))) shouldBe a[MeshUploadPlan.Gpu4D]

  "Edge endpoints" should "follow the chain, offset to the position" in:
    val edges = MeshFactory.mesh4D(tesseract(None)).get.edges.toIndexedSeq
    val nd = tesseract(Some(ProjectionSpec(5, planes = List(turn("xv", 30f))))).copy(x = 1f)
    val endpoints = TesseractEdgeSceneBuilder.edgeEndpoints(nd, edges)
    endpoints.size shouldBe edges.size
    endpoints.flatten should have size edges.size.toLong
    val view = ProjectionConversion.liftedProjection(nd.projectionN.get)
    val (v0, v1) = edges.head
    val (start, end) = endpoints.head.get
    (start.x, start.y, start.z) shouldBe (view(v0).x + 1f, view(v0).y, view(v0).z)
    (end.x, end.y, end.z) shouldBe (view(v1).x + 1f, view(v1).y, view(v1).z)

  it should "equal the 4D edges for a 4D ProjectionSpec" in:
    val edges = MeshFactory.mesh4D(tesseract(None)).get.edges.toIndexedSeq
    val legacy = tesseract(Some(Projection4DSpec(rotXW = 0f, rotYW = 0f, rotZW = 35f)))
    val nd = tesseract(Some(ProjectionSpec(4, planes = List(turn("zw", 35f)))))
    val expected = TesseractEdgeSceneBuilder.edgeEndpoints(legacy, edges)
    val actual = TesseractEdgeSceneBuilder.edgeEndpoints(nd, edges)
    expected.zip(actual).foreach { (e, a) =>
      val (e0, e1) = e.get
      val (a0, a1) = a.get
      Seq(e0.x - a0.x, e0.y - a0.y, e0.z - a0.z, e1.x - a1.x).foreach(_.abs should be < Tolerance)
    }

  it should "leave out an edge with an endpoint at or behind a stage's eye" in:
    val edges = MeshFactory.mesh4D(tesseract(None)).get.edges.toIndexedSeq
    val close = tesseract(Some(
      ProjectionSpec(5, planes = List(turn("xv", 90f)), eye = 0.6f, screen = 0.5f)
    ))
    val endpoints = TesseractEdgeSceneBuilder.edgeEndpoints(close, edges)
    endpoints.count(_.isEmpty) should be > 0

  "The scene builders" should "refuse a ProjectionSpec where only a 4D shader exists" in:
    val ifs = ObjectSpec(
      "menger4d", level = Some(1f), projection = Some(ProjectionSpec(5))
    )
    val message = Instanced4DSceneBuilder(IFS4DType.Menger4D).validate(List(ifs), 10)
    message.left.toOption.get should include ("menger4d cannot take a ProjectionSpec")
    message.left.toOption.get should include ("mesh type")
    LSystemSceneBuilder().validate(
      List(ObjectSpec("lsystem", projection = Some(ProjectionSpec(5)))), 10
    ).isLeft shouldBe true

  it should "refuse a fractional-level sponge with a ProjectionSpec, naming the type and level" in:
    val fractional = ObjectSpec(
      "tesseract-sponge", level = Some(1.5f), projection = Some(ProjectionSpec(5))
    )
    val result = TriangleMeshSceneBuilder(".").validate(List(fractional), 10)
    result.left.toOption.get should include ("tesseract-sponge with a ProjectionSpec")
    result.left.toOption.get should include ("whole level")
    TriangleMeshSceneBuilder(".").validate(
      List(fractional.copy(level = Some(2f))), 10
    ) shouldBe Right(())

  it should "still accept a fractional-level sponge with the 4D view" in:
    val legacy = ObjectSpec("tesseract-sponge", level = Some(1.5f))
    TriangleMeshSceneBuilder(".").validate(List(legacy), 10) shouldBe Right(())
