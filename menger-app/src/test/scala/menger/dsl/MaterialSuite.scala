package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MaterialSuite extends AnyFlatSpec with Matchers:

  "Material" should "be constructible with default parameters" in:
    val m = Material()
    m.color shouldBe Color.White
    m.ior shouldBe 1f
    m.roughness shouldBe 0.5f
    m.metallic shouldBe 0f
    m.specular shouldBe 0.5f
    m.emission shouldBe 0f

  it should "be constructible with custom parameters" in:
    val m = Material(
      color = Color.Red,
      ior = 1.5f,
      roughness = 0.3f,
      metallic = 0.8f,
      specular = 0.9f,
      emission = 0.1f
    )
    m.color shouldBe Color.Red
    m.ior shouldBe 1.5f
    m.roughness shouldBe 0.3f
    m.metallic shouldBe 0.8f

  it should "validate parameter ranges" in:
    an[IllegalArgumentException] should be thrownBy Material(ior = -1f)
    an[IllegalArgumentException] should be thrownBy Material(roughness = 1.5f)
    an[IllegalArgumentException] should be thrownBy Material(metallic = -0.1f)
    an[IllegalArgumentException] should be thrownBy Material(specular = 1.1f)
    an[IllegalArgumentException] should be thrownBy Material(emission = -0.1f)

  "Material presets" should "have correct Glass properties" in:
    val glass = Material.Glass
    glass.ior shouldBe 1.5f
    glass.roughness shouldBe 0f
    glass.metallic shouldBe 0f
    glass.specular shouldBe 1f
    glass.color.a should be < (0.1f)

  it should "have correct Water properties" in:
    val water = Material.Water
    water.ior shouldBe 1.33f +- 0.01f
    water.roughness shouldBe 0f
    water.metallic shouldBe 0f

  it should "have correct Diamond properties" in:
    val diamond = Material.Diamond
    diamond.ior shouldBe 2.42f +- 0.01f
    diamond.color.a should be < (0.1f)

  it should "have correct Chrome properties" in:
    val chrome = Material.Chrome
    chrome.metallic shouldBe 1f
    chrome.roughness shouldBe 0f

  it should "have correct Gold properties" in:
    val gold = Material.Gold
    gold.metallic shouldBe 1f
    gold.color.r shouldBe 1f
    gold.color.g should be > 0.8f
    gold.color.b shouldBe 0f

  it should "have correct Copper properties" in:
    val copper = Material.Copper
    copper.metallic shouldBe 1f
    copper.color.r should be > copper.color.g
    copper.color.g should be > copper.color.b

  "Material factory methods" should "create matte materials" in:
    val m = Material.matte(Color.Red)
    m.roughness shouldBe 1f
    m.metallic shouldBe 0f
    m.specular shouldBe 0f

  it should "create plastic materials" in:
    val m = Material.plastic(Color.Green)
    m.ior shouldBe 1.5f
    m.roughness shouldBe 0.3f
    m.metallic shouldBe 0f

  it should "create metal materials" in:
    val m = Material.metal(Color.Blue)
    m.metallic shouldBe 1f
    m.roughness shouldBe 0.1f

  it should "create glass materials with transparency" in:
    val m = Material.glass(Color("#FF0000"))
    m.ior shouldBe 1.5f
    m.color.a should be < (0.1f)
    m.specular shouldBe 1f

  "Material.toOptixMaterial" should "convert to optix Material" in:
    val dsl = Material(
      color = Color(1f, 0.5f, 0.25f, 0.8f),
      ior = 1.5f,
      roughness = 0.3f,
      metallic = 0.8f,
      specular = 0.9f
    )
    val optix = dsl.toOptixMaterial
    optix.ior shouldBe 1.5f
    optix.roughness shouldBe 0.3f
    optix.metallic shouldBe 0.8f
    optix.specular shouldBe 0.9f
    optix.color.r shouldBe 1f
    optix.color.g shouldBe 0.5f

  it should "preserve color alpha in conversion" in:
    val dsl = Material.Glass
    val optix = dsl.toOptixMaterial
    optix.color.a should be < (0.1f)
