package menger.engines.scene

import menger.dsl.ResourceLimits
import menger.objects.SpongeByVolume
import menger.spec.ObjectSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// menger#179: the triangle budget a scene is judged against is a parameter; the fixed
// ResourceLimits budget is only its default.
class SceneCostBudgetSpec extends AnyFlatSpec with Matchers:

  private val volume5 = ObjectSpec(objectType = "sponge-volume", level = Some(5f))
  private val twoVolume5 = List(volume5, volume5)
  private val twoVolume5Triangles = 2 * SpongeByVolume.triangleCount(5)

  private def check(specs: List[ObjectSpec], budget: Long) =
    SceneCost.check(SceneCost.estimate(specs, 1, SceneCost.CausticsLoad.None, budget))

  "SceneCost.estimate" should "default to the fixed budget" in:
    SceneCost.estimate(twoVolume5, 1, SceneCost.CausticsLoad.None).triangleBudget shouldBe
      ResourceLimits.sceneMaxTriangles

  it should "admit what a larger budget takes and refuse what a smaller one does not" in:
    twoVolume5Triangles should be > ResourceLimits.sceneMaxTriangles
    check(twoVolume5, ResourceLimits.sceneMaxTriangles).isLeft shouldBe true
    check(twoVolume5, twoVolume5Triangles).isRight shouldBe true
    check(List(volume5), SpongeByVolume.triangleCount(5) - 1).isLeft shouldBe true

  "SceneCost.check" should "name the budget the scene was judged against" in:
    val refusal = check(twoVolume5, 20_000_000L).left.getOrElse("")
    refusal should include(s"$twoVolume5Triangles triangles in total, budget 20000000")

  "SceneCost.fittingLevel" should "name the level that fits the given budget" in:
    SceneCost.fittingLevel(List(volume5), budget = 1_000_000L).map(_.level) shouldBe Some(4)
    SceneCost.fittingLevel(List(volume5), budget = 100_000L).map(_.level) shouldBe Some(3)
    SceneCost.fittingLevel(twoVolume5, budget = twoVolume5Triangles) shouldBe None

  "SceneCost.checkTriangles" should "judge against the given budget" in:
    SceneCost.checkTriangles(twoVolume5).isLeft shouldBe true
    SceneCost.checkTriangles(twoVolume5, twoVolume5Triangles).isRight shouldBe true

  "ResourceLimits.ndSpongeBoundAdmits" should "scale its headroom with the budget" in:
    val bound = 100_000_000L
    ResourceLimits.ndSpongeBoundAdmits(bound) shouldBe false
    ResourceLimits.ndSpongeBoundAdmits(bound, budget = 1_000_000_000L) shouldBe true
