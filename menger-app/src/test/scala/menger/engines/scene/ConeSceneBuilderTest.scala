package menger.engines.scene

import menger.ConeGeometry
import menger.ObjectSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ConeSceneBuilderTest extends AnyFlatSpec with Matchers:

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
