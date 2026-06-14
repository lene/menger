package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.CausticsConfig
import menger.dsl.{EnvMapVideo, IBL, RenderSettings, Scene, Sphere}
import menger.engines.SceneConverter

class IBLSuite extends AnyFlatSpec with Matchers:

  // === IBL case class ===

  "IBL" should "have correct defaults" in:
    val ibl = IBL()
    ibl.strength shouldBe 1.0f
    ibl.samples  shouldBe 1

  it should "accept valid custom values" in:
    val ibl = IBL(strength = 0.5f, samples = 4)
    ibl.strength shouldBe 0.5f
    ibl.samples  shouldBe 4

  it should "reject samples outside 1–8" in:
    an[IllegalArgumentException] should be thrownBy IBL(samples = 0)
    an[IllegalArgumentException] should be thrownBy IBL(samples = 9)

  it should "reject negative strength" in:
    an[IllegalArgumentException] should be thrownBy IBL(strength = -0.1f)

  // === RenderSettings.accumulation ===

  "RenderSettings" should "default accumulation to 1" in:
    RenderSettings().accumulation shouldBe 1

  it should "accept accumulation > 1" in:
    RenderSettings(accumulation = 8).accumulation shouldBe 8

  it should "reject accumulation < 1" in:
    an[IllegalArgumentException] should be thrownBy RenderSettings(accumulation = 0)

  // === Scene.ibl default ===

  private val minimalScene = Scene(objects = List(Sphere()))

  "Scene" should "default ibl to None" in:
    minimalScene.ibl shouldBe None

  it should "accept Some(IBL())" in:
    val s = minimalScene.copy(ibl = Some(IBL(strength = 0.7f)))
    s.ibl.get.strength shouldBe 0.7f

  // === SceneConverter ===

  private val noCaustics = CausticsConfig()

  "SceneConverter" should "produce iblEnabled=false when ibl is None" in:
    val configs = SceneConverter.convert(minimalScene, noCaustics)
    configs.iblEnabled  shouldBe false
    configs.iblStrength shouldBe 1.0f
    configs.iblSamples  shouldBe 1

  it should "produce iblEnabled=true with correct values when ibl is set" in:
    val scene   = minimalScene.copy(
      envMap = Some("sky.hdr"),
      ibl    = Some(IBL(strength = 0.5f, samples = 3)),
    )
    val configs = SceneConverter.convert(scene, noCaustics)
    configs.iblEnabled  shouldBe true
    configs.iblStrength shouldBe 0.5f
    configs.iblSamples  shouldBe 3

  it should "produce iblEnabled=true when ibl is set with envMapVideo" in:
    val scene   = minimalScene.copy(
      envMapVideo = Some(EnvMapVideo("sky.mov")),
      ibl         = Some(IBL(strength = 0.5f, samples = 3)),
    )
    val configs = SceneConverter.convert(scene, noCaustics)
    configs.iblEnabled  shouldBe true
    configs.iblStrength shouldBe 0.5f
    configs.iblSamples  shouldBe 3

  it should "produce accumulationFrames=1 by default" in:
    SceneConverter.convert(minimalScene, noCaustics).accumulationFrames shouldBe 1

  it should "forward accumulation from RenderSettings" in:
    val scene   = minimalScene.copy(render = Some(RenderSettings(accumulation = 4)))
    val configs = SceneConverter.convert(scene, noCaustics)
    configs.accumulationFrames shouldBe 4

  it should "warn and produce iblEnabled=false when ibl is set but envMap is absent" in:
    // no assertion on log output; just ensure no exception and iblEnabled=false
    val scene   = minimalScene.copy(ibl = Some(IBL()))
    val configs = SceneConverter.convert(scene, noCaustics)
    configs.iblEnabled shouldBe false
