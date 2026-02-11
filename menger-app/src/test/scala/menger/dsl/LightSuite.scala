package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.language.implicitConversions

class LightSuite extends AnyFlatSpec with Matchers:

  "Directional" should "be constructible with Vec3 direction" in:
    val light = Directional(Vec3(1f, -1f, -1f))
    light.direction shouldBe Vec3(1f, -1f, -1f)
    light.intensity shouldBe 1f
    light.color shouldBe Color.White

  it should "be constructible with custom intensity" in:
    val light = Directional(Vec3(1f, 0f, 0f), intensity = 2.0f)
    light.intensity shouldBe 2.0f

  it should "be constructible with custom color" in:
    val light = Directional(Vec3(1f, 0f, 0f), intensity = 1.5f, color = Color.Red)
    light.color shouldBe Color.Red

  it should "accept Float tuple positions" in:
    val light = Directional((1f, -1f, -1f))
    light.direction shouldBe Vec3(1f, -1f, -1f)

  it should "accept Float tuple with intensity" in:
    val light = Directional((1f, -1f, -1f), 2.0f)
    light.intensity shouldBe 2.0f

  it should "accept Float tuple with intensity and color" in:
    val light = Directional((1f, -1f, -1f), 1.5f, Color.Blue)
    light.color shouldBe Color.Blue

  it should "accept Int tuple positions" in:
    val light = Directional((1, -1, -1))
    light.direction shouldBe Vec3(1f, -1f, -1f)

  it should "accept Int tuple with intensity" in:
    val light = Directional((1, -1, -1), 2.0f)
    light.intensity shouldBe 2.0f

  it should "accept Double tuple positions" in:
    val light = Directional((1.0, -1.0, -1.0))
    light.direction shouldBe Vec3(1f, -1f, -1f)

  it should "validate non-negative intensity" in:
    an[IllegalArgumentException] should be thrownBy Directional(Vec3.UnitX, intensity = -1f)

  "Directional.toCommonLight" should "convert to common Light" in:
    val dsl = Directional(Vec3(1f, -1f, 0f), intensity = 2.0f, color = Color.Red)
    val common = dsl.toCommonLight

    common shouldBe a[menger.common.Light.Directional]
    common match
      case directional: menger.common.Light.Directional =>
        directional.direction(0) shouldBe 1f
        directional.direction(1) shouldBe -1f
        directional.direction(2) shouldBe 0f
        directional.intensity shouldBe 2.0f
        directional.color shouldBe Color.Red.toCommonColor
      case _ => fail("Expected Directional light")

  "Point" should "be constructible with Vec3 position" in:
    val light = Point(Vec3(0f, 5f, 0f))
    light.position shouldBe Vec3(0f, 5f, 0f)
    light.intensity shouldBe 1f
    light.color shouldBe Color.White

  it should "be constructible with custom intensity" in:
    val light = Point(Vec3(0f, 5f, 0f), intensity = 3.0f)
    light.intensity shouldBe 3.0f

  it should "be constructible with custom color" in:
    val light = Point(Vec3(0f, 5f, 0f), intensity = 1.5f, color = Color("#FFFFCC"))
    light.color shouldBe Color("#FFFFCC")

  it should "accept Float tuple positions" in:
    val light = Point((0f, 5f, 0f))
    light.position shouldBe Vec3(0f, 5f, 0f)

  it should "accept Float tuple with intensity" in:
    val light = Point((0f, 5f, 0f), 2.0f)
    light.intensity shouldBe 2.0f

  it should "accept Float tuple with intensity and color" in:
    val light = Point((0f, 5f, 0f), 1.5f, Color.Green)
    light.color shouldBe Color.Green

  it should "accept Int tuple positions" in:
    val light = Point((0, 5, 0))
    light.position shouldBe Vec3(0f, 5f, 0f)

  it should "accept Int tuple with intensity" in:
    val light = Point((0, 5, 0), 2.0f)
    light.intensity shouldBe 2.0f

  it should "accept Double tuple positions" in:
    val light = Point((0.0, 5.0, 0.0))
    light.position shouldBe Vec3(0f, 5f, 0f)

  it should "validate non-negative intensity" in:
    an[IllegalArgumentException] should be thrownBy Point(Vec3.Zero, intensity = -1f)

  "Point.toCommonLight" should "convert to common Light" in:
    val dsl = Point(Vec3(0f, 5f, 0f), intensity = 1.5f, color = Color("#FFFFCC"))
    val common = dsl.toCommonLight

    common shouldBe a[menger.common.Light.Point]
    common match
      case point: menger.common.Light.Point =>
        point.position(0) shouldBe 0f
        point.position(1) shouldBe 5f
        point.position(2) shouldBe 0f
        point.intensity shouldBe 1.5f
        point.color shouldBe Color("#FFFFCC").toCommonColor
      case _ => fail("Expected Point light")
