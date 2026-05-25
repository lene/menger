package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.dsl.{IBL, RenderSettings, Scene, Sphere}

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
