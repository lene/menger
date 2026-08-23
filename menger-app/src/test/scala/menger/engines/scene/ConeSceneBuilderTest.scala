package menger.engines.scene

import menger.ConeGeometry
import menger.ObjectRotation
import menger.ObjectSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ConeSceneBuilderTest extends AnyFlatSpec with Matchers:

  private val Tolerance = 1e-4f

  private def near(actual: (Float, Float, Float), expected: (Float, Float, Float)): Unit =
    actual._1 shouldBe expected._1 +- Tolerance
    actual._2 shouldBe expected._2 +- Tolerance
    actual._3 shouldBe expected._3 +- Tolerance

  "ConeSceneBuilder.validate" should "accept a valid cone spec" in:
    val spec = ObjectSpec(
      objectType = "cone",
      cone = ConeGeometry(apex = Some((0f, 0.5f, 0f)), base = Some((0f, -0.5f, 0f)))
    )
    val builder = ConeSceneBuilder()
    builder.validate(List(spec), 64) shouldBe Right(())

  it should "reject non-cone specs" in:
    val spec = ObjectSpec(objectType = "sphere")
    val builder = ConeSceneBuilder()
    builder.validate(List(spec), 64) shouldBe a[Left[?, ?]]

  it should "reject empty spec list" in:
    val builder = ConeSceneBuilder()
    builder.validate(List.empty, 64) shouldBe a[Left[?, ?]]

  it should "reject when too many instances" in:
    val specs = List.fill(5)(ObjectSpec(objectType = "cone"))
    val builder = ConeSceneBuilder()
    builder.validate(specs, 3) shouldBe a[Left[?, ?]]

  "ConeSceneBuilder.isCompatible" should "return true for two cones" in:
    val spec1 = ObjectSpec(objectType = "cone")
    val spec2 = ObjectSpec(objectType = "cone")
    ConeSceneBuilder().isCompatible(spec1, spec2) shouldBe true

  it should "return false for mixed types" in:
    val cone = ObjectSpec(objectType = "cone")
    val sphere = ObjectSpec(objectType = "sphere")
    ConeSceneBuilder().isCompatible(cone, sphere) shouldBe false

  "ConeSceneBuilder.calculateInstanceCount" should "return the number of specs" in:
    val specs = List.fill(3)(ObjectSpec(objectType = "cone"))
    ConeSceneBuilder().calculateInstanceCount(specs) shouldBe 3L

  // Sprint 36 H5.2: a cone with no explicit apex/base is placed by rotating a local offset
  // and translating by spec position -- rotation= was previously ignored entirely for this
  // (the overwhelmingly common) case, since the default apex/base derivation used pos+size
  // with no rotation applied at all.
  "ConeSceneBuilder.rotatedWorldPoint" should "reduce to plain pos+offset at zero rotation" in:
    val spec = ObjectSpec(objectType = "cone", x = 1f, y = 2f, z = 3f)
    near(
      ConeSceneBuilder().rotatedWorldPoint(spec, 0f, 0.5f, 0f),
      (1f, 2.5f, 3f)
    )

  it should "rotate the local offset 90 degrees about X before translating" in:
    // Rotating local (0, 0.5, 0) by 90 deg about X sends +Y to +Z.
    val spec = ObjectSpec(
      objectType = "cone", x = 1f, y = 2f, z = 3f,
      rotation = ObjectRotation(x = (math.Pi / 2).toFloat)
    )
    near(
      ConeSceneBuilder().rotatedWorldPoint(spec, 0f, 0.5f, 0f),
      (1f, 2f, 3.5f)
    )

  it should "rotate the local offset 90 degrees about Y before translating" in:
    // Rotating local (0, 0, 0.5) by 90 deg about Y sends +Z to +X.
    val spec = ObjectSpec(
      objectType = "cone", x = 1f, y = 2f, z = 3f,
      rotation = ObjectRotation(y = (math.Pi / 2).toFloat)
    )
    near(
      ConeSceneBuilder().rotatedWorldPoint(spec, 0f, 0f, 0.5f),
      (1.5f, 2f, 3f)
    )
