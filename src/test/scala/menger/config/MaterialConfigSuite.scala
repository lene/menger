package menger.config

import com.badlogic.gdx.graphics.Color
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


class MaterialConfigSuite extends AnyFlatSpec with Matchers:

  "MaterialConfig case class" should "have sensible defaults" in:
    val config = MaterialConfig()
    config.color shouldBe Color.WHITE
    config.ior shouldBe 1.5f

  it should "allow custom color" in:
    val customColor = new Color(0.5f, 0.3f, 0.7f, 1.0f)
    val config = MaterialConfig(color = customColor)
    config.color shouldBe customColor

  it should "allow custom ior" in:
    val config = MaterialConfig(ior = 2.42f)
    config.ior shouldBe 2.42f

  "MaterialConfig.Default" should "have same values as default constructor" in:
    MaterialConfig.Default.color shouldBe Color.WHITE
    MaterialConfig.Default.ior shouldBe 1.5f

  "MaterialConfig.Glass preset" should "have low alpha for transparency" in:
    MaterialConfig.Glass.color.a shouldBe 0.1f

  it should "have standard glass IOR (1.5)" in:
    MaterialConfig.Glass.ior shouldBe 1.5f

  "MaterialConfig.Diamond preset" should "have high IOR (2.42)" in:
    MaterialConfig.Diamond.ior shouldBe 2.42f

  it should "have white color" in:
    MaterialConfig.Diamond.color shouldBe Color.WHITE

  "MaterialConfig.Mirror preset" should "have IOR of 1.0 (no refraction)" in:
    MaterialConfig.Mirror.ior shouldBe 1.0f

  it should "have white color" in:
    MaterialConfig.Mirror.color shouldBe Color.WHITE

  "MaterialConfig.Water preset" should "have water IOR (1.33)" in:
    MaterialConfig.Water.ior shouldBe 1.33f

  it should "have blue-tinted color" in:
    MaterialConfig.Water.color.b shouldBe 1.0f
    MaterialConfig.Water.color.r shouldBe 0.8f +- 0.01f

  it should "have low alpha for transparency" in:
    MaterialConfig.Water.color.a shouldBe 0.3f +- 0.01f

  // Edge case tests
  "MaterialConfig edge cases" should "accept IOR at boundary 1.0 (vacuum/no refraction)" in:
    val config = MaterialConfig(ior = 1.0f)
    config.ior shouldBe 1.0f

  it should "accept IOR less than 1.0 (no physical validation)" in:
    val config = MaterialConfig(ior = 0.5f)
    config.ior shouldBe 0.5f

  it should "accept IOR at 0.0 (no physical validation)" in:
    val config = MaterialConfig(ior = 0.0f)
    config.ior shouldBe 0.0f

  it should "accept negative IOR (no physical validation)" in:
    val config = MaterialConfig(ior = -1.0f)
    config.ior shouldBe -1.0f

  it should "accept very large IOR" in:
    val config = MaterialConfig(ior = 100.0f)
    config.ior shouldBe 100.0f

  it should "work with fully transparent color (alpha = 0)" in:
    val config = MaterialConfig(color = new Color(1f, 1f, 1f, 0f))
    config.color.a shouldBe 0f

  it should "work with fully opaque color (alpha = 1)" in:
    val config = MaterialConfig(color = new Color(1f, 1f, 1f, 1f))
    config.color.a shouldBe 1f
