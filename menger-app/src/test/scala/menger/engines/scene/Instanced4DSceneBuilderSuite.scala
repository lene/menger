package menger.engines.scene

import io.github.lene.optix.MengerRenderer
import menger.ObjectSpec
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Covers Instanced4DSceneBuilder for all three IFS 4D types (F9: was three
  * near-identical suites — Menger4D/Sierpinski4D/Hexadecachoron4DSceneBuilderSuite —
  * now parameterized over IFS4DType.all). Coverage preserved: validate (integer and
  * fractional level, missing level, wrong type), calculateInstanceCount (integer,
  * fractional, mixed), and the native-failure path. */
class Instanced4DSceneBuilderSuite extends AnyFlatSpec with Matchers with MockFactory:

  private def spec(s: String): ObjectSpec = ObjectSpec.parse(s).toOption.get

  IFS4DType.all.foreach { ifsType =>

    s"Instanced4DSceneBuilder(${ifsType.name}).validate" should "accept integer level" in:
      val builder = Instanced4DSceneBuilder(ifsType)
      val result = builder.validate(List(spec(s"type=${ifsType.name}:level=3")), maxInstances = 10)
      result shouldBe Right(())

    it should "accept fractional level" in:
      val builder = Instanced4DSceneBuilder(ifsType)
      val fracLevel = spec(s"type=${ifsType.name}:level=1").copy(level = Some(1.5f))
      builder.validate(List(fracLevel), maxInstances = 10) shouldBe Right(())

    it should "reject missing level" in:
      val builder = Instanced4DSceneBuilder(ifsType)
      val noLevel = spec(s"type=${ifsType.name}:level=1").copy(level = None)
      builder.validate(List(noLevel), maxInstances = 10) shouldBe a[Left[String, Unit]]

    it should "reject wrong object type" in:
      val builder = Instanced4DSceneBuilder(ifsType)
      val wrongType = spec("type=sphere:level=2")
      builder.validate(List(wrongType), maxInstances = 10) shouldBe a[Left[String, Unit]]

    s"Instanced4DSceneBuilder(${ifsType.name}).calculateInstanceCount" should "return 1 for integer level" in:
      val builder = Instanced4DSceneBuilder(ifsType)
      builder.calculateInstanceCount(List(spec(s"type=${ifsType.name}:level=2"))) shouldBe 1L

    it should "return 2 for fractional level" in:
      val builder = Instanced4DSceneBuilder(ifsType)
      val fracLevel = spec(s"type=${ifsType.name}:level=1").copy(level = Some(1.5f))
      builder.calculateInstanceCount(List(fracLevel)) shouldBe 2L

    it should "return correct total for mixed integer and fractional levels" in:
      val builder = Instanced4DSceneBuilder(ifsType)
      val fracLevel = spec(s"type=${ifsType.name}:level=1").copy(level = Some(1.5f))
      val specs = List(
        spec(s"type=${ifsType.name}:level=1"),  // integer → 1
        fracLevel,                               // fractional → 2
        spec(s"type=${ifsType.name}:level=3"),  // integer → 1
      )
      builder.calculateInstanceCount(specs) shouldBe 4L
  }

  // Menger4d is the only IFS type with recursion-depth bounds (must match
  // MAX_4D_LEVEL in OptiXWrapper.cpp). Sierpinski/Hexadecachoron are unbounded.
  "Instanced4DSceneBuilder(menger4d).validate" should "reject level above max (14)" in:
    val builder = Instanced4DSceneBuilder(IFS4DType.Menger4D)
    val result = builder.validate(List(spec("type=menger4d:level=15")), maxInstances = 10)
    result shouldBe a[Left[String, Unit]]
    result.left.getOrElse("") should include("[0, 14]")

  it should "reject fractional level whose floor+1 exceeds max (14)" in:
    val builder = Instanced4DSceneBuilder(IFS4DType.Menger4D)
    val fracOver = spec("type=menger4d:level=1").copy(level = Some(14.5f))
    val result = builder.validate(List(fracOver), maxInstances = 10)
    result shouldBe a[Left[String, Unit]]

  "Instanced4DSceneBuilder(menger4d).buildScene" should "fail when native instance creation fails" in:
    val renderer = mock[MengerRenderer]
    val recorded = scala.collection.mutable.ArrayBuffer.empty[InstanceId]
    val builder = Instanced4DSceneBuilder(
      IFS4DType.Menger4D,
      recorder = (_: Int, instanceId: InstanceId) => recorded += instanceId
    )

    (renderer.addMenger4DInstance _).expects(*, *, *, *, *, *, *, *, *, *).returning(-1).once()

    val result = builder.buildScene(List(spec("type=menger4d:level=1")), renderer, maxInstances = 10)

    result.isFailure shouldBe true
    result.failed.get.getMessage should include("Native renderer failed to add menger4d instance")
    recorded shouldBe empty
