package menger.spec

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The scene-level nD view of a 4D object (menger#162): what it accepts at construction. */
class ProjectionSpecSuite extends AnyFlatSpec with Matchers:

  private def turn(plane: String, degrees: Float) =
    PlaneRotation(PlaneName(plane), Degrees(degrees))

  "ProjectionSpec" should "default to the 4D eye and screen with no rotation" in:
    val view = ProjectionSpec(5)
    view.planes shouldBe Nil
    (view.eye, view.screen) shouldBe (Projection4DSpec.DefaultEyeW, Projection4DSpec.DefaultScreenW)
    view.stageCount shouldBe 2

  it should "use the single eye and screen for every stage when none are listed" in:
    ProjectionSpec(6, eye = 4f, screen = 2f).stagePairs shouldBe
      List.fill(3)(ProjectionStage(4f, 2f))

  it should "use the listed stages, one per dimension above 3" in:
    val stages = List(ProjectionStage(5f, 2f), ProjectionStage(3f, 1f))
    ProjectionSpec(5, stages = stages).stagePairs shouldBe stages

  it should "reject a stage count other than dimension - 3, naming both counts" in:
    val message = intercept[IllegalArgumentException] {
      ProjectionSpec(5, stages = List(ProjectionStage(3f, 1.5f)))
    }.getMessage
    message should include ("dimension 5 needs 2 projection stages, got 1")

  it should "accept dimensions 4 to 8 only" in:
    (4 to 8).foreach(d => ProjectionSpec(d).dimension shouldBe d)
    Seq(3, 9, 0, -1).foreach { d =>
      intercept[IllegalArgumentException](ProjectionSpec(d)).getMessage should
        include (s"dimension must be between 4 and 8, is $d")
    }

  it should "reject a plane whose axis the dimension does not have, naming plane and dimension" in:
    val message = intercept[IllegalArgumentException] {
      ProjectionSpec(5, planes = List(turn("xs", 10f)))
    }.getMessage
    message should include ("plane 'xs'")
    message should include ("dimension 5")

  it should "accept every plane up to the dimension's last axis" in:
    ProjectionSpec(8, planes = List(turn("xs", 1f), turn("ts", 1f), turn("vu", 1f)))
      .planes.size shouldBe 3
    ProjectionSpec(5, planes = List(turn("xv", 1f), turn("vx", 1f))).planes.size shouldBe 2

  it should "reject an eye that is not beyond the screen" in:
    intercept[IllegalArgumentException](ProjectionSpec(5, eye = 1f, screen = 2f))
    intercept[IllegalArgumentException](ProjectionStage(1f, 1f))
    intercept[IllegalArgumentException](ProjectionStage(1f, -1f))

  "PlaneName" should "name two distinct known axes" in:
    PlaneName("xw").firstAxis shouldBe 0
    PlaneName("xw").secondAxis shouldBe 3
    PlaneName("sv").highestAxis shouldBe 7

  it should "reject an unknown letter, the same axis twice and a wrong length" in:
    intercept[IllegalArgumentException](PlaneName("xq")).getMessage should include ("'xq'")
    intercept[IllegalArgumentException](PlaneName("xx")).getMessage should
      include ("'xx' names the same axis twice")
    intercept[IllegalArgumentException](PlaneName("xyz"))
    intercept[IllegalArgumentException](PlaneName("x"))

  it should "convert from a string literal, so a scene writes PlaneRotation(\"xv\", 30f)" in:
    val rotation = PlaneRotation("xv", 30f)
    rotation.plane shouldBe PlaneName("xv")
    rotation.angle.value shouldBe 30f

  "ObjectSpec" should "tell a ProjectionSpec from the 4D shorthand" in:
    val legacy = ObjectSpec("tesseract", projection = Some(Projection4DSpec(rotXW = 20f)))
    val nd = ObjectSpec("tesseract", projection = Some(ProjectionSpec(5)))
    legacy.isLegacyView shouldBe true
    legacy.projection4D shouldBe Some(Projection4DSpec(rotXW = 20f))
    legacy.projectionN shouldBe None
    nd.isLegacyView shouldBe false
    nd.projection4D shouldBe None
    nd.projectionN shouldBe Some(ProjectionSpec(5))
    ObjectSpec("tesseract").isLegacyView shouldBe true

  it should "drop a ProjectionSpec with the rest of the view, but keep the w-scale" in:
    val nd = ObjectSpec("tesseract", projection = Some(ProjectionSpec(5)))
    nd.withoutView shouldBe ObjectSpec("tesseract")
    val scaled = ObjectSpec(
      "tesseract", projection = Some(Projection4DSpec(rotXW = 9f, wScale = 0.5f))
    )
    scaled.withoutView.projection shouldBe Some(Projection4DSpec(wScale = 0.5f))
