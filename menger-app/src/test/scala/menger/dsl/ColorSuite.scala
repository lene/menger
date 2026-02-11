package menger.dsl

import scala.language.implicitConversions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ColorSuite extends AnyFlatSpec with Matchers:

  "Color" should "be constructible with r, g, b components" in:
    val c = Color(1f, 0f, 0.5f)
    c.r shouldBe 1f
    c.g shouldBe 0f
    c.b shouldBe 0.5f
    c.a shouldBe 1f

  it should "be constructible with r, g, b, a components" in:
    val c = Color(1f, 0f, 0.5f, 0.8f)
    c.a shouldBe 0.8f

  it should "provide common color constants" in:
    Color.White shouldBe Color(1f, 1f, 1f)
    Color.Black shouldBe Color(0f, 0f, 0f)
    Color.Red shouldBe Color(1f, 0f, 0f)
    Color.Green shouldBe Color(0f, 1f, 0f)
    Color.Blue shouldBe Color(0f, 0f, 1f)

  it should "reject invalid component values" in:
    an[IllegalArgumentException] should be thrownBy Color(1.5f, 0f, 0f)
    an[IllegalArgumentException] should be thrownBy Color(0f, -0.1f, 0f)
    an[IllegalArgumentException] should be thrownBy Color(0f, 0f, 0f, 1.1f)

  "Color hex parsing" should "parse 6-character hex strings" in:
    Color("#FF0000") shouldBe Color(1f, 0f, 0f)
    Color("#00FF00") shouldBe Color(0f, 1f, 0f)
    Color("#0000FF") shouldBe Color(0f, 0f, 1f)
    val gray = Color("#808080")
    gray.r shouldBe 0.5f +- 0.01f
    gray.g shouldBe 0.5f +- 0.01f
    gray.b shouldBe 0.5f +- 0.01f

  it should "parse hex strings without # prefix" in:
    Color("FF0000") shouldBe Color(1f, 0f, 0f)
    Color("00FF00") shouldBe Color(0f, 1f, 0f)

  it should "parse 8-character hex strings with alpha" in:
    val c = Color("#FF000080")
    c.r shouldBe 1f
    c.g shouldBe 0f
    c.b shouldBe 0f
    c.a shouldBe 0.5f +- 0.01f

  it should "be case-insensitive" in:
    Color("#ff0000") shouldBe Color("#FF0000")
    Color("#AbCdEf") shouldBe Color("#ABCDEF")

  it should "reject invalid hex strings" in:
    an[IllegalArgumentException] should be thrownBy Color("#GGGGGG")
    an[IllegalArgumentException] should be thrownBy Color("#FF00")
    an[IllegalArgumentException] should be thrownBy Color("#FF00000")
    an[IllegalArgumentException] should be thrownBy Color("")

  "Color.toCommonColor" should "convert to menger.common.Color" in:
    val dslColor = Color(1f, 0.5f, 0.25f, 0.8f)
    val common = dslColor.toCommonColor
    common.r shouldBe 1f
    common.g shouldBe 0.5f
    common.b shouldBe 0.25f
    common.a shouldBe 0.8f

  "Color string conversion" should "work implicitly" in:
    val c: Color = "#FF0000"
    c shouldBe Color(1f, 0f, 0f)
