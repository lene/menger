package menger.engines.scene

import menger.ObjectSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Menger4DSceneBuilderSuite extends AnyFlatSpec with Matchers:

  private def spec(s: String): ObjectSpec = ObjectSpec.parse(s).toOption.get

  "Menger4DSceneBuilder.validate" should "accept fractional level" in:
    val builder = Menger4DSceneBuilder()
    val result = builder.validate(List(spec("type=menger4d:level=1.5")), maxInstances = 10)
    result shouldBe Right(())

  it should "reject missing level" in:
    val builder = Menger4DSceneBuilder()
    val noLevel = spec("type=menger4d:level=1").copy(level = None)
    builder.validate(List(noLevel), maxInstances = 10) shouldBe a[Left[String, Unit]]

  "Menger4DSceneBuilder.calculateInstanceCount" should "return 1 for integer level" in:
    val builder = Menger4DSceneBuilder()
    builder.calculateInstanceCount(List(spec("type=menger4d:level=2"))) shouldBe 1L

  it should "return 2 for fractional level" in:
    val builder = Menger4DSceneBuilder()
    builder.calculateInstanceCount(List(spec("type=menger4d:level=1.5"))) shouldBe 2L

  it should "return correct total for mixed integer and fractional levels" in:
    val builder = Menger4DSceneBuilder()
    val specs = List(
      spec("type=menger4d:level=1"),    // integer → 1
      spec("type=menger4d:level=1.5"),  // fractional → 2
      spec("type=menger4d:level=2"),    // integer → 1
    )
    builder.calculateInstanceCount(specs) shouldBe 4L
