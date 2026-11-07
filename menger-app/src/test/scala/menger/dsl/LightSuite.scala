package menger.dsl

import scala.language.implicitConversions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

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

  "AreaLight" should "be constructible with position, normal, and radius" in:
    val light = AreaLight(Vec3(0f, 5f, 0f), Vec3(0f, -1f, 0f), radius = 1.0f)
    light.position shouldBe Vec3(0f, 5f, 0f)
    light.normal shouldBe Vec3(0f, -1f, 0f)
    light.radius shouldBe 1.0f
    light.intensity shouldBe 1.0f
    light.color shouldBe Color.White
    light.shadowSamples shouldBe 4

  it should "be constructible with custom intensity and color" in:
    val light = AreaLight(Vec3(1f, 3f, 0f), Vec3(0f, -1f, 0f), radius = 0.5f, intensity = 2.0f, color = Color.Red)
    light.intensity shouldBe 2.0f
    light.color shouldBe Color.Red

  it should "be constructible with custom shadowSamples" in:
    val light = AreaLight(Vec3(0f, 5f, 0f), Vec3(0f, -1f, 0f), radius = 1.0f, shadowSamples = 8)
    light.shadowSamples shouldBe 8

  it should "accept Float tuple for position and normal" in:
    val light = AreaLight((0f, 5f, 0f), (0f, -1f, 0f), radius = 1.0f)
    light.position shouldBe Vec3(0f, 5f, 0f)
    light.normal shouldBe Vec3(0f, -1f, 0f)

  it should "validate non-negative intensity" in:
    an[IllegalArgumentException] should be thrownBy
      AreaLight(Vec3.Zero, Vec3.UnitY, radius = 1.0f, intensity = -0.1f)

  it should "validate positive radius" in:
    an[IllegalArgumentException] should be thrownBy
      AreaLight(Vec3.Zero, Vec3.UnitY, radius = 0f)

  it should "validate shadowSamples range lower bound" in:
    an[IllegalArgumentException] should be thrownBy
      AreaLight(Vec3.Zero, Vec3.UnitY, radius = 1.0f, shadowSamples = 0)

  it should "validate shadowSamples range upper bound" in:
    an[IllegalArgumentException] should be thrownBy
      AreaLight(Vec3.Zero, Vec3.UnitY, radius = 1.0f, shadowSamples = 17)

  "AreaLight.toCommonLight" should "convert to common Area light" in:
    val dsl = AreaLight(Vec3(0f, 5f, 0f), Vec3(0f, -1f, 0f), radius = 1.5f, intensity = 2.0f, color = Color.Blue, shadowSamples = 8)
    val common = dsl.toCommonLight

    common shouldBe a[menger.common.Light.Area]
    common match
      case area: menger.common.Light.Area =>
        area.position(0) shouldBe 0f
        area.position(1) shouldBe 5f
        area.position(2) shouldBe 0f
        area.normal(0) shouldBe 0f
        area.normal(1) shouldBe -1f
        area.normal(2) shouldBe 0f
        area.radius shouldBe 1.5f
        area.intensity shouldBe 2.0f
        area.color shouldBe Color.Blue.toCommonColor
        area.shadowSamples shouldBe 8
        area.shape shouldBe menger.common.AreaLightShape.Disk
      case _ => fail("Expected Area light")

  "Sun" should "default to the sun's apparent size, 50 units away, white, intensity 1" in:
    val sun = Sun(Vec3(0f, -1f, 0f))
    sun.diameterDegrees shouldBe 0.53f
    sun.distance shouldBe 50f
    sun.color shouldBe Color.White
    sun.intensity shouldBe 1f

  it should "accept a Float tuple for the direction" in:
    Sun((1f, -1f, -1f)).direction shouldBe Vec3(1f, -1f, -1f)

  it should "become an area light on the sun's direction" in:
    val area = Sun(Vec3(0f, -2f, 0f), diameterDegrees = 2f, distance = 10f).toAreaLight
    area.position shouldBe Vec3(0f, 10f, 0f)
    area.normal shouldBe Vec3(0f, -1f, 0f)
    area.radius shouldBe (10f * math.tan(math.toRadians(1.0)).toFloat +- 1e-6f)

  it should "place the disk against the direction the light travels" in:
    val area = Sun(Vec3(3f, -4f, 0f), distance = 5f).toAreaLight
    area.position.x shouldBe -3f +- 1e-5f
    area.position.y shouldBe 4f +- 1e-5f
    area.position.z shouldBe 0f +- 1e-5f

  it should "scale the intensity by 1 + distance^2 so the origin is lit like a Directional" in:
    List(20f, 80f).foreach { distance =>
      Sun(Vec3.UnitX, distance = distance, intensity = 2f).toAreaLight.intensity shouldBe
        (2f * (1f + distance * distance) +- 1e-3f)
    }

  it should "carry its colour to the area light" in:
    Sun(Vec3.UnitX, color = Color.Red).toAreaLight.color shouldBe Color.Red

  it should "convert to a common Area light" in:
    Sun(Vec3(0f, -1f, 0f), distance = 10f, intensity = 0.5f).toCommonLight match
      case area: menger.common.Light.Area =>
        area.position(1) shouldBe 10f
        area.normal(1) shouldBe -1f
        area.intensity shouldBe (0.5f * 101f +- 1e-3f)
      case other => fail(s"Expected an Area light, got $other")

  it should "reject a zero direction, naming the field" in:
    intercept[IllegalArgumentException](Sun(Vec3.Zero)).getMessage should
      include("Sun.direction")

  it should "reject a diameter of zero or less, or above 10 degrees, naming the field" in:
    List(0f, -1f, 10.5f).foreach { diameter =>
      intercept[IllegalArgumentException](Sun(Vec3.UnitX, diameterDegrees = diameter))
        .getMessage should include("Sun.diameterDegrees")
    }
    Sun(Vec3.UnitX, diameterDegrees = 10f).diameterDegrees shouldBe 10f

  it should "reject a distance of zero or less, naming the field" in:
    List(0f, -5f).foreach { distance =>
      intercept[IllegalArgumentException](Sun(Vec3.UnitX, distance = distance)).getMessage should
        include("Sun.distance")
    }

  it should "reject a negative intensity, naming the field" in:
    intercept[IllegalArgumentException](Sun(Vec3.UnitX, intensity = -0.1f)).getMessage should
      include("Sun.intensity")
