package menger.common

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MengerExceptionsSuite extends AnyFlatSpec with Matchers:

  "ParseException" should "format message with input" in:
    val ex = ParseException("Invalid format", "abc123")
    ex.getMessage shouldBe "Invalid format. Input: 'abc123'"

  it should "include expected format when provided" in:
    val ex = ParseException("Invalid format", "abc123", Some("number"))
    ex.getMessage shouldBe "Invalid format. Input: 'abc123'. Expected: number"

  "ValidationException" should "format message with field and value" in:
    val ex = ValidationException("Value out of range", "radius", -5)
    ex.getMessage shouldBe "Value out of range. Field: 'radius', Value: '-5'"

  it should "include constraint when provided" in:
    val ex = ValidationException("Value out of range", "radius", -5, Some("must be positive"))
    ex.getMessage shouldBe "Value out of range. Field: 'radius', Value: '-5'. Constraint: must be positive"

  "ConfigurationException" should "format simple message" in:
    val ex = ConfigurationException("Missing required setting")
    ex.getMessage shouldBe "Missing required setting"

  it should "include config key when provided" in:
    val ex = ConfigurationException("Invalid value", Some("maxInstances"))
    ex.getMessage shouldBe "Invalid value. Config key: 'maxInstances'"

  "UnknownGeometryException" should "format message with geometry type" in:
    val ex = UnknownGeometryException("invalid-type")
    ex.getMessage shouldBe "Unknown geometry type: 'invalid-type'"

  it should "include valid types when provided" in:
    val ex = UnknownGeometryException("invalid-type", Seq("cube", "sphere", "sponge"))
    ex.getMessage shouldBe "Unknown geometry type: 'invalid-type'. Valid types: cube, sphere, sponge"

  "AnimationException" should "format simple message" in:
    val ex = AnimationException("Missing frames")
    ex.getMessage shouldBe "Missing frames"

  it should "include spec index when provided" in:
    val ex = AnimationException("Missing frames", Some(2))
    ex.getMessage shouldBe "Missing frames (specification index: 2)"

  "InvalidDirectionException" should "format message with coordinates and reason" in:
    val ex = InvalidDirectionException(1, 1, 0, "must be a unit vector")
    ex.getMessage shouldBe "Invalid direction (1.0, 1.0, 0.0): must be a unit vector"

  "All exceptions" should "extend MengerException" in:
    ParseException("test", "input") shouldBe a[MengerException]
    ValidationException("test", "field", "value") shouldBe a[MengerException]
    ConfigurationException("test") shouldBe a[MengerException]
    UnknownGeometryException("type") shouldBe a[MengerException]
    AnimationException("test") shouldBe a[MengerException]
    InvalidDirectionException(0, 0, 0) shouldBe a[MengerException]

  they should "extend RuntimeException" in:
    ParseException("test", "input") shouldBe a[RuntimeException]
    ValidationException("test", "field", "value") shouldBe a[RuntimeException]
    ConfigurationException("test") shouldBe a[RuntimeException]
