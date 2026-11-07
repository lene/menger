package menger.dsl

import menger.engines.SceneConverter
import menger.spec.ObjectType
import menger.spec.PlaneName
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The DSL's n-cube (menger#163): dimension 4..8, a ProjectionSpec of at least that dimension,
  * and the spec the renderer gets. */
class HypercubeSuite extends AnyFlatSpec with Matchers:

  "Hypercube" should "be constructible from its dimension alone" in:
    val cube = Hypercube(5)
    cube.dimension shouldBe 5
    cube.pos shouldBe Vec3.Zero
    cube.size shouldBe 1.0f
    cube.projection shouldBe None

  it should "accept every dimension from 4 to 8" in:
    (4 to 8).foreach(n => Hypercube(n).toObjectSpec.dimension shouldBe Some(n))

  it should "reject a dimension outside 4..8, naming the range" in:
    Seq(-1, 0, 3, 9, 12).foreach { n =>
      val refusal = the[IllegalArgumentException] thrownBy Hypercube(n)
      refusal.getMessage should (include("Hypercube.dimension") and include("4-8"))
    }

  it should "reject a projection of fewer dimensions than the cube, naming both" in:
    val refusal = the[IllegalArgumentException] thrownBy
      Hypercube(6, projection = Some(ProjectionSpec(5)))
    refusal.getMessage should (include("6-cube") and include("dimension = 5"))

  it should "accept a projection of the cube's dimension or more" in:
    Hypercube(5, projection = Some(ProjectionSpec(5))).projection should not be empty
    Hypercube(5, projection = Some(ProjectionSpec(7))).projection should not be empty

  it should "validate positive size and non-negative ior" in:
    an[IllegalArgumentException] should be thrownBy Hypercube(5, size = 0f)
    an[IllegalArgumentException] should be thrownBy Hypercube(5, ior = -1f)

  "Hypercube.toObjectSpec" should "name the n-cube type and carry the dimension" in:
    val spec = Hypercube(6, size = 2f).toObjectSpec
    spec.objectType shouldBe "n-cube"
    ObjectType.isNCube(spec.objectType) shouldBe true
    (spec.dimension, spec.size) shouldBe ((Some(6), 2f))

  it should "default the projection to ProjectionSpec(N)" in:
    Hypercube(6).toObjectSpec.projectionN shouldBe Some(ProjectionSpec(6))

  it should "carry the given projection" in:
    val view = ProjectionSpec(7, planes = List(PlaneRotation(PlaneName("xt"), 20f)))
    Hypercube(5, projection = Some(view)).toObjectSpec.projectionN shouldBe Some(view)

  it should "carry edges, material and colour" in:
    val spec = Hypercube(
      5, material = Some(Material.Film), edgeRadius = Some(0.03f),
      edgeMaterial = Some(Material.Gold), color = Some(Color(1f, 0f, 0f))
    ).toObjectSpec
    spec.edgeRadius shouldBe Some(0.03f)
    spec.edgeMaterial should not be empty
    spec.material should not be empty
    spec.hasEdgeRendering shouldBe true

  it should "survive the scene conversion with its dimension and view" in:
    val view = ProjectionSpec(6, planes = List(PlaneRotation(PlaneName("xu"), 20f)))
    val scene = Scene(Camera.Default, Hypercube(5, projection = Some(view)))
    val specs = SceneConverter.convert(scene, Caustics.Disabled.toCausticsConfig)
      .scene.objectSpecs.getOrElse(Nil)
    specs.map(s => (s.objectType, s.dimension, s.projectionN)) shouldBe
      List(("n-cube", Some(5), Some(view)))

  "The manifest bound" should "agree with the constructor's" in:
    FieldConstraints.fields(("Hypercube", "dimension")).admits(4d) shouldBe true
    FieldConstraints.fields(("Hypercube", "dimension")).admits(8d) shouldBe true
    FieldConstraints.fields(("Hypercube", "dimension")).admits(9d) shouldBe false
