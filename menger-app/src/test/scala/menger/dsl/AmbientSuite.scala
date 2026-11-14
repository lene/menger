package menger.dsl

import org.scalatest.EitherValues
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AmbientSuite extends AnyFlatSpec with Matchers with EitherValues:

  "Ambient.Occlusion" should "default to optix-jni's own default: 16 samples, unlimited radius" in:
    val occlusion = Ambient.Occlusion()
    occlusion.samples shouldBe 16
    occlusion.radius shouldBe Float.PositiveInfinity

  it should "accept the ends of its ranges" in:
    noException should be thrownBy Ambient.Occlusion(samples = 1, radius = 0.01f)
    noException should be thrownBy Ambient.Occlusion(samples = 64, radius = Float.PositiveInfinity)

  it should "reject a sample count outside 1-64, naming the field, the range and the value" in:
    val tooFew = the[IllegalArgumentException] thrownBy Ambient.Occlusion(samples = 0)
    tooFew.getMessage should (include("samples") and include("1-64") and include("0"))
    val tooMany = the[IllegalArgumentException] thrownBy Ambient.Occlusion(samples = 65)
    tooMany.getMessage should (include("samples") and include("1-64") and include("65"))

  it should "reject a radius below 0.01, naming the field, the bound and the value" in:
    val tooSmall = the[IllegalArgumentException] thrownBy Ambient.Occlusion(radius = 0.005f)
    tooSmall.getMessage should (include("radius") and include("0.01") and include("0.005"))
    an[IllegalArgumentException] should be thrownBy Ambient.Occlusion(radius = 0f)
    an[IllegalArgumentException] should be thrownBy Ambient.Occlusion(radius = -1f)
    an[IllegalArgumentException] should be thrownBy Ambient.Occlusion(radius = Float.NaN)

  "Ambient.parse" should "read constant" in:
    Ambient.parse("constant").value shouldBe Ambient.Constant

  it should "read occlusion with its defaults" in:
    Ambient.parse("occlusion").value shouldBe Ambient.Occlusion()

  it should "read occlusion with a sample count" in:
    Ambient.parse("occlusion:32").value shouldBe Ambient.Occlusion(samples = 32)

  it should "read occlusion with a sample count and a radius" in:
    Ambient.parse("occlusion:64:0.5").value shouldBe Ambient.Occlusion(64, 0.5f)

  it should "read inf as the unlimited radius" in:
    Ambient.parse("occlusion:16:inf").value shouldBe Ambient.Occlusion(16, Float.PositiveInfinity)

  it should "ignore case and surrounding blanks" in:
    Ambient.parse("  Occlusion:8:INF ").value shouldBe Ambient.Occlusion(8, Float.PositiveInfinity)

  it should "read an empty sample count as the default" in:
    Ambient.parse("occlusion::2").value shouldBe Ambient.Occlusion(radius = 2f)

  it should "reject an unknown model, naming the value and the choices" in:
    val message = Ambient.parse("fog").left.value
    message should (include("fog") and include("constant") and include("occlusion"))

  it should "reject a sample count out of range, naming the value and the range" in:
    Ambient.parse("occlusion:0").left.value should (include("0") and include("1-64"))
    Ambient.parse("occlusion:65").left.value should (include("65") and include("1-64"))

  it should "reject a radius out of range, naming the value and the bound" in:
    Ambient.parse("occlusion:16:0.001").left.value should (include("0.001") and include("0.01"))
    Ambient.parse("occlusion:16:-2").left.value should (include("-2") and include("0.01"))

  it should "reject a value that is no number, naming it" in:
    Ambient.parse("occlusion:many").left.value should include("many")
    Ambient.parse("occlusion:16:wide").left.value should include("wide")

  it should "reject extra parts and parameters on constant" in:
    Ambient.parse("occlusion:16:1:9").left.value should include("occlusion:16:1:9")
    Ambient.parse("constant:16").left.value should include("constant:16")

  "Ambient.describe" should "write a model the way --ambient reads it" in:
    Ambient.describe(Ambient.Constant) shouldBe "constant"
    Ambient.describe(Ambient.Occlusion(16, Float.PositiveInfinity)) shouldBe "occlusion:16:inf"
    Ambient.describe(Ambient.Occlusion(64, 0.5f)) shouldBe "occlusion:64:0.5"
