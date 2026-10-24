package menger.engines.scene

import menger.ObjectSpec
import menger.common.Material
import menger.dsl.RenderSettings
import menger.dsl.ResourceLimits
import menger.dsl.HexadecachoronGasket
import menger.dsl.Menger4D
import menger.dsl.PentachoronGasket
import menger.objects.SpongeByVolume
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// Review 2026-10-14, menger#80/#106: each field had its own cap, the product had none.
class SceneCostSpec extends AnyFlatSpec with Matchers:

  private def volumeSponge(level: Float, material: Option[Material] = None) =
    ObjectSpec(objectType = "sponge-volume", level = Some(level), material = material)

  private def fits(specs: List[ObjectSpec], accumulation: Int = 1,
                   caustics: SceneCost.CausticsLoad = SceneCost.CausticsLoad.None) =
    SceneCost.check(SceneCost.estimate(specs, accumulation, caustics)).isRight

  "SceneCost" should "admit one level-5 volume sponge but not two" in:
    fits(List(volumeSponge(5f))) shouldBe true
    fits(List(volumeSponge(5f), volumeSponge(5f))) shouldBe false

  it should "count both whole levels of a fractional level" in:
    SceneCost.triangles(volumeSponge(4.5f)) shouldBe
      SpongeByVolume.triangleCount(4) + SpongeByVolume.triangleCount(5)

  it should "multiply level-mix worlds by accumulation" in:
    val glass = (1 to 4).toList.map(_ => volumeSponge(1.5f, Some(Material.Glass)))
    SceneCost.rendersPerFrame(glass, 1) shouldBe 16
    SceneCost.rendersPerFrame(glass :+ volumeSponge(1.5f, Some(Material.Glass)), 1) shouldBe 6
    SceneCost.rendersPerFrame(glass, 8) shouldBe 128
    fits(glass, accumulation = 8) shouldBe true
    fits(glass, accumulation = 9) shouldBe false

  it should "bound caustics photons per frame" in:
    fits(List(volumeSponge(1f)), caustics = SceneCost.CausticsLoad(500_000L, 20L)) shouldBe true
    fits(List(volumeSponge(1f)), caustics = SceneCost.CausticsLoad(10_000_000L, 1000L)) shouldBe
      false

  // menger#132: refused from the level, before a level-4 mesh is built to count its edges.
  it should "admit edge tubes on a sponge up to level 3, not 3.5 or 4" in:
    def edged(level: Float) = volumeSponge(level).copy(edgeRadius = Some(0.01f))
    fits(List(edged(3f))) shouldBe true
    fits(List(edged(3.5f))) shouldBe false
    fits(List(edged(4f))) shouldBe false
    fits(List(volumeSponge(4f))) shouldBe true

  it should "name every exceeded budget" in:
    val over = SceneCost.Estimate(ResourceLimits.sceneMaxTriangles + 1,
      ResourceLimits.sceneMaxRendersPerFrame + 1, ResourceLimits.sceneMaxPhotonsPerFrame + 1,
      ResourceLimits.edgeTubeSpongeMaxLevel + 1)
    val message = SceneCost.check(over).left.getOrElse("")
    message should (include("triangles") and include("renders per frame") and include("photons")
      and include("edge tubes"))

  // Session 6, msa#48: the rejection names a level that fits, not only the totals.
  it should "name the highest level at which the most expensive object fits" in:
    val specs = List(volumeSponge(5f), volumeSponge(5.8f))
    val fit = SceneCost.fittingLevel(specs)
    fit.map(_.objectType) shouldBe Some("sponge-volume")
    fit.map(_.level) shouldBe Some(4)
    fit.map(_.sceneTriangles) shouldBe
      Some(SpongeByVolume.triangleCount(5) + SpongeByVolume.triangleCount(4))
    SceneCost.checkTriangles(specs).left.getOrElse("") should
      include("the sponge-volume fits at level 4")

  it should "name no level for a scene within the budget" in:
    SceneCost.fittingLevel(List(volumeSponge(5f))) shouldBe None

  "RenderSettings.accumulation" should "be capped" in:
    noException should be thrownBy RenderSettings(accumulation = ResourceLimits.renderAccumulationMax)
    an[IllegalArgumentException] should be thrownBy
      RenderSettings(accumulation = ResourceLimits.renderAccumulationMax + 1)

  "The 4D IFS fractals" should "reject a level above the 4D IFS ceiling" in:
    val tooHigh = ResourceLimits.ifs4dMaxLevel + 1f
    an[IllegalArgumentException] should be thrownBy PentachoronGasket(level = tooHigh)
    an[IllegalArgumentException] should be thrownBy Menger4D(level = tooHigh)
    an[IllegalArgumentException] should be thrownBy HexadecachoronGasket(level = tooHigh)
