package menger.engines

import menger.app.GlobalRotation
import menger.cli.SafeMengerCLIOptions
import menger.engines.scene.TrackedMesh4D
import menger.spec.ObjectSpec
import menger.spec.PlaneName
import menger.spec.PlaneRotation
import menger.spec.Projection4DSpec
import menger.spec.ProjectionSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** A `ProjectionSpec` has no fast path, no window rotation and no CLI rotation (menger#162):
  * everything that rewrites or updates a 4D view in place leaves it alone or rebuilds. */
class NdProjectionGuardsSuite extends AnyFlatSpec with Matchers:

  private def view(degrees: Float) =
    ProjectionSpec(5, planes = List(PlaneRotation(PlaneName("xv"), degrees)))

  private val legacy = ObjectSpec("tesseract", projection = Some(Projection4DSpec(rotXW = 10f)))
  private val nd = ObjectSpec("tesseract", projection = Some(view(30f)))
  private val ndTurned = ObjectSpec("tesseract", projection = Some(view(31f)))

  "specsDifferAtMostIn4DProjection" should "stay true for a changed 4D view" in:
    val turned = ObjectSpec("tesseract", projection = Some(Projection4DSpec(rotXW = 20f)))
    WithAnimation.specsDifferAtMostIn4DProjection(List(legacy), List(turned)) shouldBe true

  it should "be false when a ProjectionSpec changes, so the frame is rebuilt" in:
    WithAnimation.specsDifferAtMostIn4DProjection(List(nd), List(ndTurned)) shouldBe false

  it should "be false between a ProjectionSpec and a 4D view of the same object" in:
    WithAnimation.specsDifferAtMostIn4DProjection(List(nd), List(legacy)) shouldBe false
    WithAnimation.specsDifferAtMostIn4DProjection(List(legacy), List(nd)) shouldBe false

  it should "be true for an unchanged ProjectionSpec: nothing to update" in:
    WithAnimation.specsDifferAtMostIn4DProjection(List(nd), List(nd)) shouldBe true

  "TrackedMesh4D.canUpdateInPlace" should "refuse a ProjectionSpec on either side" in:
    TrackedMesh4D.canUpdateInPlace(List(nd), List(ndTurned)) shouldBe false
    TrackedMesh4D.canUpdateInPlace(List(nd), List(nd)) shouldBe false
    TrackedMesh4D.canUpdateInPlace(List(legacy), List(nd)) shouldBe false

  it should "still take a changed 4D view" in:
    val turned = ObjectSpec("tesseract", projection = Some(Projection4DSpec(rotXW = 20f)))
    TrackedMesh4D.canUpdateInPlace(List(legacy), List(turned)) shouldBe true

  "Interactive4DOffset.applyTo" should "rotate a 4D view and leave a ProjectionSpec as it is" in:
    val offset = Interactive4DOffset().plus(5f, 0f, 0f)
    val applied = offset.applyTo(List(legacy, nd))
    applied.head.projection4D.map(_.rotXW.value) shouldBe Some(15f)
    applied(1) shouldBe nd

  it should "report the file's ProjectionSpec as a projection that a reload can change" in:
    Interactive4DOffset.fileProjections(List(nd)) should not be
      Interactive4DOffset.fileProjections(List(ndTurned))

  "GlobalRotation" should "leave a ProjectionSpec as it is" in:
    val opts = SafeMengerCLIOptions(
      List("--objects", "type=tesseract", "--rot-x-w", "30")
    )
    GlobalRotation(opts, List(nd)).head.projection shouldBe nd.projection

  "Window4DState" should "describe a 4D view and say nothing about a ProjectionSpec" in:
    Window4DState.describe(List(legacy)) should not be empty
    Window4DState.describe(List(nd)) shouldBe None
    Window4DState.describe(List(nd, legacy)).get should include ("xw 10.0")
