package menger.engines.scene

import menger.ObjectSpec
import menger.common.Color
import menger.common.Material
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MaterialExtractorSuite extends AnyFlatSpec with Matchers:

  private val gold = Material(Color(1f, 0.84f, 0f), roughness = 0.2f, metallic = 1f)
  private val tint = Color(1f, 0f, 0f)

  "MaterialExtractor.extract" should "use the material as-is when no color is given" in:
    val spec = ObjectSpec(objectType = "sphere", material = Some(gold))
    MaterialExtractor.extract(spec) shouldBe gold

  it should "tint the material's color when both material and color are given (H5.3)" in:
    val spec = ObjectSpec(objectType = "sphere", material = Some(gold), color = Some(tint))
    val extracted = MaterialExtractor.extract(spec)
    extracted.color shouldBe tint
    // every other property still comes from the material, not the default
    extracted.roughness shouldBe gold.roughness
    extracted.metallic shouldBe gold.metallic

  it should "use color directly when no material is given" in:
    val spec = ObjectSpec(objectType = "sphere", color = Some(tint))
    MaterialExtractor.extract(spec).color shouldBe tint

  // Usability review 2026-09 (F14): an explicit `color` on a transparent material must not
  // silently make it opaque -- `color` tints hue, transparency stays owned by the material.
  it should "preserve the material's alpha when an opaque color is given (F14)" in:
    val glass = Material.Glass
    val opaqueTint = Color(0.2f, 0.5f, 0.9f, 1f)
    val spec = ObjectSpec(objectType = "sphere", material = Some(glass), color = Some(opaqueTint))
    val extracted = MaterialExtractor.extract(spec)
    extracted.color.r shouldBe opaqueTint.r
    extracted.color.g shouldBe opaqueTint.g
    extracted.color.b shouldBe opaqueTint.b
    extracted.color.a shouldBe glass.color.a
    extracted.ior shouldBe glass.ior
