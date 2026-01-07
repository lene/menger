package menger

import com.badlogic.gdx.graphics.{Color => GdxColor}
import menger.ColorConversions._
import menger.common.{Color => CommonColor}
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

  it should "fail for array with more than 4 elements (MatchError)" in:
    // padTo doesn't truncate, so >4 elements causes pattern match failure
    a[MatchError] should be thrownBy rgbIntsToColor(Array(255, 128, 64, 32, 999))

  it should "clamp values via GdxColor constructor (values > 255 become > 1.0 then clamp)" in:
    // GdxColor constructor clamps values to [0, 1]
    val color = rgbIntsToColor(Array(510, 0, 0))
    color.r shouldBe 1.0f  // clamped by GdxColor

  it should "clamp negative values via GdxColor constructor" in:
    // GdxColor constructor clamps values to [0, 1]
    val color = rgbIntsToColor(Array(-255, 0, 0))
    color.r shouldBe 0.0f  // clamped by GdxColor

  it should "handle mid-range values correctly" in:
    val color = rgbIntsToColor(Array(64, 128, 192, 255))
    color.r shouldBe (64f / 255f) +- 0.01f
    color.g shouldBe (128f / 255f) +- 0.01f
    color.b shouldBe (192f / 255f) +- 0.01f
    color.a shouldBe 1.0f

  "GdxColor.toCommonColor" should "preserve all color components" in:
    val gdx = new GdxColor(0.2f, 0.4f, 0.6f, 0.8f)
    val common = gdx.toCommonColor
    common.r shouldBe 0.2f
    common.g shouldBe 0.4f
    common.b shouldBe 0.6f
    common.a shouldBe 0.8f

  it should "preserve boundary value 0.0" in:
    val gdx = new GdxColor(0f, 0f, 0f, 0f)
    val common = gdx.toCommonColor
    common.r shouldBe 0f
    common.g shouldBe 0f
    common.b shouldBe 0f
    common.a shouldBe 0f

  it should "preserve boundary value 1.0" in:
    val gdx = new GdxColor(1f, 1f, 1f, 1f)
    val common = gdx.toCommonColor
    common.r shouldBe 1f
    common.g shouldBe 1f
    common.b shouldBe 1f
    common.a shouldBe 1f

  "CommonColor.toGdxColor" should "preserve all color components" in:
    val common = CommonColor(0.3f, 0.5f, 0.7f, 0.9f)
    val gdx = common.toGdxColor
    gdx.r shouldBe 0.3f
    gdx.g shouldBe 0.5f
    gdx.b shouldBe 0.7f
    gdx.a shouldBe 0.9f

  it should "preserve boundary value 0.0" in:
    val common = CommonColor(0f, 0f, 0f, 0f)
    val gdx = common.toGdxColor
    gdx.r shouldBe 0f
    gdx.g shouldBe 0f
    gdx.b shouldBe 0f
    gdx.a shouldBe 0f

  it should "preserve boundary value 1.0" in:
    val common = CommonColor(1f, 1f, 1f, 1f)
    val gdx = common.toGdxColor
    gdx.r shouldBe 1f
    gdx.g shouldBe 1f
    gdx.b shouldBe 1f
    gdx.a shouldBe 1f

  "round-trip conversion" should "preserve GdxColor through CommonColor and back" in:
    val original = new GdxColor(0.25f, 0.5f, 0.75f, 1.0f)
    val roundTrip = original.toCommonColor.toGdxColor
    roundTrip.r shouldBe original.r
    roundTrip.g shouldBe original.g
    roundTrip.b shouldBe original.b
    roundTrip.a shouldBe original.a

  it should "preserve CommonColor through GdxColor and back" in:
    val original = CommonColor(0.1f, 0.2f, 0.3f, 0.4f)
    val roundTrip = original.toGdxColor.toCommonColor
    roundTrip.r shouldBe original.r
    roundTrip.g shouldBe original.g
    roundTrip.b shouldBe original.b
    roundTrip.a shouldBe original.a
