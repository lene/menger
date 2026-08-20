package menger

import menger.cli.SafeMengerCLIOptions
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GlobalRotationTest extends AnyFlatSpec with Matchers:

  private val Tolerance = 1e-4f

  // Sprint 36 H5.1: --rot-x/-y/-z/-x-w/-y-w/-z-w were parsed but never applied anywhere --
  // RotationProjectionParameters.apply(opts), their sole reader, had no caller.

  "GlobalRotation" should "leave specs untouched when no global rotation is given" in:
    val opts = SafeMengerCLIOptions(List("--objects", "type=cube"))
    val specs = List(ObjectSpec(objectType = "cube"))
    val result = GlobalRotation(opts, specs)
    result.head.rotation shouldBe ObjectRotation()

  it should "add --rot-x/-y/-z (degrees, converted to radians) onto a 3D object's rotation" in:
    val opts = SafeMengerCLIOptions(List("--objects", "type=cube", "--rot-x", "90", "--rot-y", "45"))
    val specs = List(ObjectSpec(objectType = "cube"))
    val result = GlobalRotation(opts, specs)
    result.head.rotation.x shouldBe (math.Pi / 2).toFloat +- Tolerance
    result.head.rotation.y shouldBe (math.Pi / 4).toFloat +- Tolerance
    result.head.rotation.z shouldBe 0f +- Tolerance

  it should "compose with a per-object rot-x= already set on the spec" in:
    val opts = SafeMengerCLIOptions(List("--objects", "type=cube:rot-x=30", "--rot-x", "60"))
    val specs = List(ObjectSpec(objectType = "cube", rotation = ObjectRotation(x = (math.Pi / 6).toFloat)))
    val result = GlobalRotation(opts, specs)
    result.head.rotation.x shouldBe (math.Pi / 2).toFloat +- Tolerance

  private val zeroProjection = Projection4DSpec(rotXW = 0f, rotYW = 0f, rotZW = 0f)

  it should "leave a 4D fast-path spec's projection4D untouched when no global 4D rotation is given" in:
    val opts = SafeMengerCLIOptions(List("--objects", "type=menger4d:level=1"))
    val specs = List(ObjectSpec(objectType = "menger4d", projection4D = Some(zeroProjection)))
    val result = GlobalRotation(opts, specs)
    result.head.projection4D shouldBe Some(zeroProjection)

  it should "add --rot-x-w/-y-w/-z-w (degrees, no conversion) onto a 4D fast-path spec's projection4D" in:
    val opts = SafeMengerCLIOptions(
      List("--objects", "type=menger4d:level=1", "--rot-x-w", "30", "--rot-y-w", "15")
    )
    val specs = List(ObjectSpec(objectType = "menger4d", projection4D = Some(zeroProjection)))
    val result = GlobalRotation(opts, specs)
    result.head.projection4D.map(_.rotXW) shouldBe Some(30f)
    result.head.projection4D.map(_.rotYW) shouldBe Some(15f)
    result.head.projection4D.map(_.rotZW) shouldBe Some(0f)

  it should "not touch projection4D of a plain 3D object even if global 4D rotation is given" in:
    val opts = SafeMengerCLIOptions(List("--objects", "type=cube", "--rot-x-w", "30"))
    val specs = List(ObjectSpec(objectType = "cube"))
    val result = GlobalRotation(opts, specs)
    result.head.projection4D shouldBe None
