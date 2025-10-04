package menger.objects

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SpongeByVolumeSuite extends AnyFlatSpec with Matchers:

  "calculateAlphaForFractionalLevel with original alpha = 1.0" should "return 1.0 when fractional part is 0" in:
    val material = new Material(ColorAttribute.createDiffuse(new Color(1f, 1f, 1f, 1f)))
    val alpha = FractionalLevelSponge.calculateAlphaForFractionalLevel(material, 0f)
    alpha shouldBe 1.0f

  it should "return 0.5 when fractional part is 0.5" in:
    val material = new Material(ColorAttribute.createDiffuse(new Color(1f, 1f, 1f, 1f)))
    val alpha = FractionalLevelSponge.calculateAlphaForFractionalLevel(material, 0.5f)
    alpha shouldBe 0.5f +- 0.001f

  it should "return ~0 when fractional part approaches 1" in:
    val material = new Material(ColorAttribute.createDiffuse(new Color(1f, 1f, 1f, 1f)))
    val alpha = FractionalLevelSponge.calculateAlphaForFractionalLevel(material, 0.99f)
    alpha shouldBe 0.01f +- 0.001f

  "calculateAlphaForFractionalLevel with original alpha = 0.8" should "return 0.8 when fractional part is 0" in:
    val material = new Material(ColorAttribute.createDiffuse(new Color(1f, 1f, 1f, 0.8f)))
    val alpha = FractionalLevelSponge.calculateAlphaForFractionalLevel(material, 0f)
    alpha shouldBe 0.8f +- 0.001f

  it should "return 0.4 when fractional part is 0.5" in:
    val material = new Material(ColorAttribute.createDiffuse(new Color(1f, 1f, 1f, 0.8f)))
    val alpha = FractionalLevelSponge.calculateAlphaForFractionalLevel(material, 0.5f)
    alpha shouldBe 0.4f +- 0.001f

  it should "return ~0 when fractional part approaches 1" in:
    val material = new Material(ColorAttribute.createDiffuse(new Color(1f, 1f, 1f, 0.8f)))
    val alpha = FractionalLevelSponge.calculateAlphaForFractionalLevel(material, 0.99f)
    alpha shouldBe 0.008f +- 0.001f
