package menger.engines.scene

import menger.ObjectSpec
import menger.common.Material
import menger.dsl.RenderSettings
import menger.dsl.ResourceLimits
import menger.dsl.Sierpinski4D
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

  it should "name every exceeded budget" in:
    val over = SceneCost.Estimate(ResourceLimits.sceneMaxTriangles + 1,
      ResourceLimits.sceneMaxRendersPerFrame + 1, ResourceLimits.sceneMaxPhotonsPerFrame + 1)
    val message = SceneCost.check(over).left.getOrElse("")
    message should (include("triangles") and include("renders per frame") and include("photons"))

  "RenderSettings.accumulation" should "be capped" in:
    noException should be thrownBy RenderSettings(accumulation = ResourceLimits.renderAccumulationMax)
    an[IllegalArgumentException] should be thrownBy
      RenderSettings(accumulation = ResourceLimits.renderAccumulationMax + 1)

  "Sierpinski4D" should "reject a level above the 4D IFS ceiling" in:
    an[IllegalArgumentException] should be thrownBy
      Sierpinski4D(level = ResourceLimits.ifs4dMaxLevel + 1f)
