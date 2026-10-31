package menger.spec

import scala.language.implicitConversions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AnglesSuite extends AnyFlatSpec with Matchers:

  private val Tolerance = 1e-5f

  "Degrees" should "convert to radians" in:
    Degrees(180f).toRadians.value shouldBe math.Pi.toFloat +- Tolerance
    Degrees(90f).toRadians.value shouldBe (math.Pi / 2).toFloat +- Tolerance
    Degrees.Zero.toRadians.value shouldBe 0f

  it should "add, subtract, negate and scale" in:
    (Degrees(30f) + Degrees(15f)).value shouldBe 45f
    (Degrees(30f) - Degrees(15f)).value shouldBe 15f
    (-Degrees(30f)).value shouldBe -30f
    (Degrees(30f) * 2f).value shouldBe 60f

  "Radians" should "convert to degrees" in:
    Radians(math.Pi.toFloat).toDegrees.value shouldBe 180f +- 1e-3f
    Radians.Zero.toDegrees.value shouldBe 0f

  it should "round-trip through degrees" in:
    Degrees(37.5f).toRadians.toDegrees.value shouldBe 37.5f +- 1e-3f

  "A plain Float" should "be accepted where an angle is expected, so scene text keeps compiling" in:
    val projection = Projection4DSpec(rotXW = 30f, rotYW = 45f, rotZW = 60f)
    projection.rotXW.value shouldBe 30f
    projection.rotYW.value shouldBe 45f
    projection.rotZW.value shouldBe 60f
    ObjectRotation(0.5f, 1.0f, 1.5f).y.value shouldBe 1.0f

  it should "become an angle with .deg and .rad" in:
    90f.deg.value shouldBe 90f
    1.5f.rad.value shouldBe 1.5f

  "The 4D defaults" should "be stated in degrees" in:
    Projection4DSpec.DefaultRotXW.value shouldBe 15f
    Projection4DSpec.DefaultRotYW.value shouldBe 10f
    Projection4DSpec.DefaultRotZW shouldBe Degrees.Zero
