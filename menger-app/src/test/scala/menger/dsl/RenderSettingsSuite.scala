package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RenderSettingsSuite extends AnyFlatSpec with Matchers:

  "RenderSettings" should "have correct defaults" in:
    val rs = RenderSettings()
    rs.shadows shouldBe true
    rs.transparentShadows shouldBe false
    rs.antialiasing shouldBe false
    rs.aaMaxDepth shouldBe 2
    rs.aaThreshold shouldBe 0.1f
    rs.maxRayDepth shouldBe None
    rs.denoise shouldBe DenoiseMode.Off

  it should "be constructible with all fields" in:
    val rs = RenderSettings(
      shadows = true,
      transparentShadows = true,
      antialiasing = true,
      aaMaxDepth = 3,
      aaThreshold = 0.05f,
      denoise = DenoiseMode.Final
    )
    rs.shadows shouldBe true
    rs.transparentShadows shouldBe true
    rs.antialiasing shouldBe true
    rs.aaMaxDepth shouldBe 3
    rs.aaThreshold shouldBe 0.05f
    rs.denoise shouldBe DenoiseMode.Final

  it should "validate aaMaxDepth range" in:
    an[IllegalArgumentException] should be thrownBy RenderSettings(aaMaxDepth = 0)
    an[IllegalArgumentException] should be thrownBy RenderSettings(aaMaxDepth = 5)

  it should "validate aaThreshold range" in:
    an[IllegalArgumentException] should be thrownBy RenderSettings(aaThreshold = -0.1f)
    an[IllegalArgumentException] should be thrownBy RenderSettings(aaThreshold = 1.1f)

  it should "reject maxRayDepth out of range" in:
    an[IllegalArgumentException] should be thrownBy RenderSettings(maxRayDepth = Some(0))
    an[IllegalArgumentException] should be thrownBy RenderSettings(maxRayDepth = Some(100))

  "RenderSettings.toRenderConfig" should "map all fields to RenderConfig" in:
    val rs = RenderSettings(
      shadows = true,
      transparentShadows = true,
      antialiasing = true,
      aaMaxDepth = 3,
      aaThreshold = 0.05f
    )
    val config = rs.toRenderConfig
    config.shadows shouldBe true
    config.transparentShadows shouldBe true
    config.antialiasing shouldBe true
    config.aaMaxDepth shouldBe 3
    config.aaThreshold shouldBe 0.05f

  it should "pass maxRayDepth through to RenderConfig" in:
    val config = RenderSettings(maxRayDepth = Some(3)).toRenderConfig
    config.maxRayDepth shouldBe 3

  it should "use RenderConfig default for maxRayDepth when None" in:
    val config = RenderSettings().toRenderConfig
    config.maxRayDepth shouldBe menger.common.RenderConfig.Default.maxRayDepth

  it should "map defaults correctly" in:
    val config = RenderSettings().toRenderConfig
    config.shadows shouldBe true
    config.transparentShadows shouldBe false
    config.antialiasing shouldBe false
    config.aaMaxDepth shouldBe 2
    config.aaThreshold shouldBe 0.1f

  "RenderSettings.Default" should "match no-arg constructor" in:
    RenderSettings.Default shouldBe RenderSettings()

  "RenderSettings.HighQuality" should "have elevated settings" in:
    val hq = RenderSettings.HighQuality
    hq.shadows shouldBe true
    hq.antialiasing shouldBe true
    hq.aaMaxDepth shouldBe 3
    hq.aaThreshold shouldBe 0.05f
    hq.denoise shouldBe DenoiseMode.Off
