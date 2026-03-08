package menger.dsl

import org.scalacheck.Gen
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class MaterialValidationSuite extends AnyFlatSpec with Matchers with ScalaCheckPropertyChecks:

  "Material.validate()" should "warn on IOR below 1.0" in:
    Material(ior = 0.5f).validate() should not be empty

  it should "warn on metallic + high IOR" in:
    Material(metallic = 1.0f, ior = 1.5f).validate() should not be empty

  it should "pass for physically valid glass" in:
    Material(ior = 1.5f, roughness = 0.01f, metallic = 0.0f).validate() shouldBe empty

  it should "pass for chrome-like values (metallic=1.0, IOR=1.0)" in:
    Material(ior = 1.0f, roughness = 0.0f, metallic = 1.0f).validate() shouldBe empty

  it should "pass for partial metallic (0.5) with IOR=1.0" in:
    Material(ior = 1.0f, metallic = 0.5f).validate() shouldBe empty

  it should "warn on metallic with emission" in:
    Material(metallic = 1.0f, emission = 1.0f).validate() should not be empty

  it should "warn on high roughness with metallic" in:
    Material(roughness = 0.95f, metallic = 0.8f).validate() should not be empty

  it should "warn on filmThickness with metallic" in:
    Material(filmThickness = 100.0f, metallic = 0.8f).validate() should not be empty

  it should "pass for default material" in:
    Material().validate() shouldBe empty

  it should "pass for Material.Glass preset" in:
    Material.Glass.validate() shouldBe empty

  it should "pass for Material.Chrome preset" in:
    Material.Chrome.validate() shouldBe empty

  "Valid dielectric materials" should "produce no warnings" in:
    forAll(Gen.choose(1.0f, 3.0f), Gen.choose(0.0f, 1.0f)) { (ior, roughness) =>
      Material(ior = ior, roughness = roughness, metallic = 0.0f).validate() shouldBe empty
    }
