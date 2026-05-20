package menger.engines.scene

import menger.ObjectSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Sierpinski4DSceneBuilderSuite extends AnyFlatSpec with Matchers:

  private def spec(s: String): ObjectSpec = ObjectSpec.parse(s).toOption.get

  "Sierpinski4DSceneBuilder.validate" should "accept integer level" in:
    val builder = Sierpinski4DSceneBuilder()
    val result = builder.validate(List(spec("type=sierpinski4d:level=3")), maxInstances = 10)
    result shouldBe Right(())

  it should "reject missing level" in:
    val builder = Sierpinski4DSceneBuilder()
    val noLevel = spec("type=sierpinski4d:level=1").copy(level = None)
    builder.validate(List(noLevel), maxInstances = 10) shouldBe a[Left[String, Unit]]

  it should "accept fractional level" in:
    val builder = Sierpinski4DSceneBuilder()
    val fracLevel = spec("type=sierpinski4d:level=1").copy(level = Some(1.5f))
    builder.validate(List(fracLevel), maxInstances = 10) shouldBe Right(())

  it should "reject wrong object type" in:
    val builder = Sierpinski4DSceneBuilder()
    val wrongType = spec("type=menger4d:level=2")
    builder.validate(List(wrongType), maxInstances = 10) shouldBe a[Left[String, Unit]]

  "Sierpinski4DSceneBuilder.calculateInstanceCount" should "return 1 for integer level" in:
    val builder = Sierpinski4DSceneBuilder()
    builder.calculateInstanceCount(List(spec("type=sierpinski4d:level=2"))) shouldBe 1L

  it should "return 2 for fractional level" in:
    val builder = Sierpinski4DSceneBuilder()
    val fracLevel = spec("type=sierpinski4d:level=1").copy(level = Some(1.5f))
    builder.calculateInstanceCount(List(fracLevel)) shouldBe 2L

  it should "return correct total for mixed integer and fractional levels" in:
    val builder = Sierpinski4DSceneBuilder()
    val fracLevel = spec("type=sierpinski4d:level=1").copy(level = Some(1.5f))
    val specs = List(
      spec("type=sierpinski4d:level=1"),  // integer → 1
      fracLevel,                           // fractional → 2
      spec("type=sierpinski4d:level=3"),  // integer → 1
    )
    builder.calculateInstanceCount(specs) shouldBe 4L
