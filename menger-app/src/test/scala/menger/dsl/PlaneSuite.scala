package menger.dsl

import scala.language.implicitConversions

import menger.cli.Axis
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PlaneSuite extends AnyFlatSpec with Matchers:

  "AxisHelper" should "create AxisPosition with 'at' syntax" in:
    val pos = Y at -2
    pos.axis shouldBe Axis.Y
    pos.positive shouldBe false
    pos.value shouldBe -2f

  it should "handle positive values" in:
    val pos = Y at 5
    pos.axis shouldBe Axis.Y
    pos.positive shouldBe true
    pos.value shouldBe 5f

  it should "handle zero as positive" in:
    val pos = Y at 0
    pos.axis shouldBe Axis.Y
    pos.positive shouldBe true
    pos.value shouldBe 0f

  it should "work with X axis" in:
    val pos = X at 3
    pos.axis shouldBe Axis.X
    pos.value shouldBe 3f

  it should "work with Z axis" in:
    val pos = Z at -1.5
    pos.axis shouldBe Axis.Z
    pos.value shouldBe -1.5f

  it should "accept Int values" in:
    val pos = Y at 10
    pos.value shouldBe 10f

  it should "accept Double values" in:
    val pos = Y at 2.5
    pos.value shouldBe 2.5f

  "Plane" should "create solid-colored plane" in:
    val plane = Plane(Y at -2, color = Color.White)

    plane.axisPosition.axis shouldBe Axis.Y
    plane.axisPosition.value shouldBe -2f
    plane.color shouldBe Some(Color.White)
    plane.checkered shouldBe None

  it should "create solid-colored plane from hex string" in:
    val plane = Plane(Y at -2, "#808080")

    plane.color.isDefined shouldBe true
    val c = plane.color.get
    c.r shouldBe 0.5f +- 0.01f
    c.g shouldBe 0.5f +- 0.01f
    c.b shouldBe 0.5f +- 0.01f

  it should "create checkered plane" in:
    val plane = Plane.checkered(Y at -2, (Color.White, Color.Black))

    plane.color shouldBe None
    plane.checkered shouldBe Some((Color.White, Color.Black))

  it should "create checkered plane from hex strings" in:
    val plane = Plane.checkered(Y at -2, "#FFFFFF", "#000000")

    plane.checkered.isDefined shouldBe true
    val (c1, c2) = plane.checkered.get
    c1 shouldBe Color.White
    c2 shouldBe Color.Black

  it should "reject plane with neither color nor checkered" in:
    an[IllegalArgumentException] should be thrownBy:
      Plane(Y at -2, None, None)

  it should "reject plane with both color and checkered" in:
    an[IllegalArgumentException] should be thrownBy:
      Plane(Y at -2, Some(Color.White), Some((Color.White, Color.Black)))

  "Plane.toPlaneSpec" should "convert solid plane correctly" in:
    val plane = Plane(Y at -2, Color.White)
    val spec = plane.toPlaneSpec

    spec.axis shouldBe Axis.Y
    spec.positive shouldBe false
    spec.value shouldBe -2f

  it should "convert plane at positive position" in:
    val plane = Plane(X at 5, Color.Red)
    val spec = plane.toPlaneSpec

    spec.axis shouldBe Axis.X
    spec.positive shouldBe true
    spec.value shouldBe 5f

  it should "convert plane at zero" in:
    val plane = Plane(Z at 0, Color.Blue)
    val spec = plane.toPlaneSpec

    spec.axis shouldBe Axis.Z
    spec.positive shouldBe true
    spec.value shouldBe 0f

  "Plane.toPlaneColorSpec" should "convert solid color correctly" in:
    val plane = Plane(Y at -2, Color.White)
    val colorSpec = plane.toPlaneColorSpec

    colorSpec.isSolid shouldBe true
    colorSpec.isCheckered shouldBe false
    colorSpec.color1.r shouldBe 1f
    colorSpec.color1.g shouldBe 1f
    colorSpec.color1.b shouldBe 1f
    colorSpec.color2 shouldBe None

  it should "convert checkered pattern correctly" in:
    val plane = Plane.checkered(Y at -2, (Color.White, Color.Black))
    val colorSpec = plane.toPlaneColorSpec

    colorSpec.isSolid shouldBe false
    colorSpec.isCheckered shouldBe true
    colorSpec.color1 shouldBe Color.White.toCommonColor
    colorSpec.color2 shouldBe Some(Color.Black.toCommonColor)

  it should "convert custom colors correctly" in:
    val customColor = Color(0.5f, 0.3f, 0.7f)
    val plane = Plane(Y at -1, customColor)
    val colorSpec = plane.toPlaneColorSpec

    colorSpec.color1.r shouldBe 0.5f +- 0.01f
    colorSpec.color1.g shouldBe 0.3f +- 0.01f
    colorSpec.color1.b shouldBe 0.7f +- 0.01f

  "Axis helpers" should "be distinct objects" in:
    X should not be Y
    Y should not be Z
    Z should not be X

  "AxisPosition" should "preserve axis information" in:
    val posY = Y at -2
    val posX = X at 3
    val posZ = Z at -1

    posY.axis shouldBe Axis.Y
    posX.axis shouldBe Axis.X
    posZ.axis shouldBe Axis.Z

  "Plane examples from documentation" should "compile and work correctly" in:
    // Solid gray floor at y = -2
    val solidPlane = Plane(Y at -2, color = "#808080")
    solidPlane.toPlaneSpec.value shouldBe -2f
    solidPlane.toPlaneColorSpec.isSolid shouldBe true

    // Checkered floor (white and black)
    val checkeredPlane1 = Plane.checkered(Y at -2, (Color.White, Color.Black))
    checkeredPlane1.toPlaneColorSpec.isCheckered shouldBe true

    // Using hex colors for checkered pattern
    val checkeredPlane2 = Plane.checkered(Y at -2, "#FFFFFF", "#000000")
    checkeredPlane2.toPlaneColorSpec.isCheckered shouldBe true
