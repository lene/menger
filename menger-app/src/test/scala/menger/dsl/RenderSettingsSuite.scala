package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RenderSettingsSuite extends AnyFlatSpec with Matchers:

  "RenderSettings" should "have correct defaults" in:
    val rs = RenderSettings()
    rs.shadows shouldBe true
    rs.transparentShadows shouldBe TransparentShadows.Auto
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
    rs.transparentShadows shouldBe TransparentShadows.On
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

  // Usability review 2026-10, session 5 (F111, menger#128): coloured glass, grey shadow.
  "TransparentShadows.Auto" should "turn coloured shadows on for a coloured transparent object" in:
    RenderSettings().toRenderConfigFor(colouredTransparent = true).transparentShadows shouldBe true
    RenderSettings().toRenderConfigFor(colouredTransparent = false).transparentShadows shouldBe false
    RenderSettings(transparentShadows = false)
      .toRenderConfigFor(colouredTransparent = true).transparentShadows shouldBe false

  "SceneConverter.isColouredTransparent" should "take tinted or patterned glass, not clear glass" in:
    import menger.engines.SceneConverter.isColouredTransparent
    def glass(colour: Option[Color], procedural: Int = 0) =
      Sphere(material = Some(Material.Glass), color = colour, proceduralType = procedural)
        .toObjectSpec
    isColouredTransparent(glass(None)) shouldBe false
    isColouredTransparent(glass(Some(Color(1f, 0.1f, 0.1f, 0.02f)))) shouldBe true
    isColouredTransparent(glass(None, procedural = 11)) shouldBe true
    isColouredTransparent(Sphere(color = Some(Color(1f, 0f, 0f))).toObjectSpec) shouldBe false

  // Decided by user 2026-10-23: "caustics should always imply transparent shadows".
  "SceneConverter" should "turn transparent shadows on whenever caustics are" in:
    import menger.engines.SceneConverter
    val caustics = menger.common.CausticsConfig(enabled = true)
    val scene = Scene(objects = List(Sphere()))
    SceneConverter.convert(scene, caustics).render.map(_.transparentShadows) shouldBe Some(true)
    val off = scene.copy(render = Some(RenderSettings(transparentShadows = false)))
    SceneConverter.convert(off, caustics).render.map(_.transparentShadows) shouldBe Some(true)
    val noCaustics = menger.common.CausticsConfig()
    SceneConverter.convert(off, noCaustics).render.map(_.transparentShadows) shouldBe Some(false)

  "RenderSettings.Default" should "match no-arg constructor" in:
    RenderSettings.Default shouldBe RenderSettings()

  "RenderSettings.HighQuality" should "have elevated settings" in:
    val hq = RenderSettings.HighQuality
    hq.shadows shouldBe true
    hq.antialiasing shouldBe true
    hq.aaMaxDepth shouldBe 3
    hq.aaThreshold shouldBe 0.05f
    hq.denoise shouldBe DenoiseMode.Off
