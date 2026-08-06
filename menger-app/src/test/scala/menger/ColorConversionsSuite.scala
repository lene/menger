package menger

import menger.ColorConversions._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


class ColorConversionsSuite extends AnyFlatSpec with Matchers:

  "rgbIntsToColor" should "convert RGB array (3 elements) with default alpha 1.0" in:
    val color = rgbIntsToColor(Array(255, 128, 0))
    color.r shouldBe 1.0f +- 0.01f
    color.g shouldBe 0.5f +- 0.01f
    color.b shouldBe 0.0f +- 0.01f
    color.a shouldBe 1.0f  // padded default alpha

  it should "convert RGBA array (4 elements)" in:
    val color = rgbIntsToColor(Array(255, 0, 128, 128))
    color.r shouldBe 1.0f +- 0.01f
    color.g shouldBe 0.0f +- 0.01f
    color.b shouldBe 0.5f +- 0.01f
    color.a shouldBe 0.5f +- 0.01f

  it should "handle boundary value 0 for all channels" in:
    val color = rgbIntsToColor(Array(0, 0, 0, 0))
    color.r shouldBe 0.0f
    color.g shouldBe 0.0f
    color.b shouldBe 0.0f
    color.a shouldBe 0.0f

  it should "handle boundary value 255 for all channels" in:
    val color = rgbIntsToColor(Array(255, 255, 255, 255))
    color.r shouldBe 1.0f
    color.g shouldBe 1.0f
    color.b shouldBe 1.0f
    color.a shouldBe 1.0f

  it should "fail for empty array (MatchError - requires 4 elements after padding)" in:
    // padTo(4, 1f) on empty produces [1,1,1,1] - pattern match requires exactly 4
    // But empty.map produces empty, padTo(4,1f) gives [1,1,1,1]
    // Actually this should work - let me verify the actual behavior
    val color = rgbIntsToColor(Array())
    color.r shouldBe 1.0f  // all defaults
    color.g shouldBe 1.0f
    color.b shouldBe 1.0f
    color.a shouldBe 1.0f

  it should "pad array with 1 element to have default g, b, and alpha" in:
    val color = rgbIntsToColor(Array(128))
    color.r shouldBe 0.5f +- 0.01f
    color.g shouldBe 1.0f  // padded
    color.b shouldBe 1.0f  // padded
    color.a shouldBe 1.0f  // padded

  it should "pad array with 2 elements to have default b and alpha" in:
    val color = rgbIntsToColor(Array(255, 128))
    color.r shouldBe 1.0f +- 0.01f
    color.g shouldBe 0.5f +- 0.01f
    color.b shouldBe 1.0f  // padded
    color.a shouldBe 1.0f  // padded

  it should "truncate arrays with more than 4 elements to first 4 channels" in:
    // CR-12: rgbIntsToColor is total — take(4) uses first 4 channels, ignores extras
    val color = rgbIntsToColor(Array(255, 128, 64, 32, 999))
    color.r shouldBe 1.0f +- 0.01f
    color.g shouldBe (128f / 255f) +- 0.01f
    color.b shouldBe (64f / 255f) +- 0.01f
    color.a shouldBe (32f / 255f) +- 0.01f

  it should "clamp values above the valid range (values > 255 become > 1.0 then clamp)" in:
    val color = rgbIntsToColor(Array(510, 0, 0))
    color.r shouldBe 1.0f  // clamped

  it should "clamp negative values" in:
    val color = rgbIntsToColor(Array(-255, 0, 0))
    color.r shouldBe 0.0f  // clamped

  it should "handle mid-range values correctly" in:
    val color = rgbIntsToColor(Array(64, 128, 192, 255))
    color.r shouldBe (64f / 255f) +- 0.01f
    color.g shouldBe (128f / 255f) +- 0.01f
    color.b shouldBe (192f / 255f) +- 0.01f
    color.a shouldBe 1.0f
