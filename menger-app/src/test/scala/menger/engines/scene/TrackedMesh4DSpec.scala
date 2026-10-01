package menger.engines.scene

import menger.ObjectRotation
import menger.ObjectSpec
import menger.Projection4DSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Usability review 2026-09, session 2 (F52): the preview window rebuilt the whole 4D sponge
  * every frame of an animation, although for every level in [n, n+1) the geometry is the same
  * (level n+1 plus level n's hole caps); only the projection and the caps' alpha change. */
class TrackedMesh4DSpec extends AnyFlatSpec with Matchers:

  private def sponge(level: Float): ObjectSpec =
    ObjectSpec.parse(s"type=tesseract-sponge:level=$level:size=1.5:material=glass")
      .fold(e => fail(e), identity)

  private def rotated(spec: ObjectSpec, rotXW: Float): ObjectSpec =
    spec.copy(projection4D = Some(Projection4DSpec(rotXW = rotXW)))

  "TrackedMesh4D.canUpdateInPlace" should "accept a projection-only change" in:
    TrackedMesh4D.canUpdateInPlace(List(sponge(2f)), List(rotated(sponge(2f), 40f))) shouldBe true

  it should "accept a level change within the same integer interval" in:
    TrackedMesh4D.canUpdateInPlace(
      List(sponge(2.3f)), List(rotated(sponge(2.7f), 40f))
    ) shouldBe true

  it should "accept a position and 3D rotation change" in:
    val base = sponge(2.3f)
    TrackedMesh4D.canUpdateInPlace(
      List(base), List(base.copy(x = 1f, z = -2f, rotation = ObjectRotation(0f, 45f, 10f)))
    ) shouldBe true

  it should "reject a level change into another interval" in:
    TrackedMesh4D.canUpdateInPlace(List(sponge(2.7f)), List(sponge(3.1f))) shouldBe false

  it should "reject a change between a fractional and an integer level" in:
    TrackedMesh4D.canUpdateInPlace(List(sponge(2.5f)), List(sponge(3f))) shouldBe false
    TrackedMesh4D.canUpdateInPlace(List(sponge(2f)), List(sponge(2.5f))) shouldBe false

  it should "reject any other change" in:
    val base = sponge(2.3f)
    TrackedMesh4D.canUpdateInPlace(List(base), List(base.copy(size = 2f))) shouldBe false
    TrackedMesh4D.canUpdateInPlace(List(base), List(base, base)) shouldBe false
    TrackedMesh4D.canUpdateInPlace(
      List(base), List(ObjectSpec.parse("type=tesseract-sponge:level=2.3:size=1.5:material=chrome")
        .fold(e => fail(e), identity))
    ) shouldBe false
