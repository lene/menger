package menger.engines.scene

import menger.common.ConfigurationException
import menger.common.Material
import menger.dsl.HexadecachoronGasket
import menger.dsl.Menger4D
import menger.dsl.PentachoronGasket
import menger.dsl.RenderSettings
import menger.dsl.ResourceLimits
import menger.objects.SpongeByVolume
import menger.objects.higher_d.NdSponge
import menger.objects.higher_d.NdSpongeVariant
import menger.spec.ObjectSpec
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

  // menger#166: an n-sponge's triangles are two per face. The volume's faces are counted
  // exactly; the surface's cost is the bound of its faces below the budget, and above it, within
  // the bound's headroom, the faces counted on keys (menger#183: nothing is built).
  private def ndSponge(
    variant: NdSpongeVariant, n: Int, level: Float, k: Int = 1
  ): ObjectSpec =
    ObjectSpec(
      objectType = variant.objectTypeName,
      level = Some(level), dimension = Some(n), maxCentreCoordinates = Some(k)
    )

  private val TrianglesPerFace = ResourceLimits.TrianglesPerFace
  private val Budget = ResourceLimits.sceneMaxTriangles

  "SceneCost for an n-sponge" should "cost its bound of faces when that is within the budget" in:
    val spec = ndSponge(NdSpongeVariant.Surface, 5, 2f)
    val bound = NdSponge.upperBoundFaces(NdSpongeVariant.Surface, 5, 2, 1)
    bound * TrianglesPerFace should be <= Budget
    SceneCost.triangles(spec) shouldBe TrianglesPerFace * bound
    SceneCost.triangles(spec) should be >= TrianglesPerFace * 28160L

  it should "count the faces exactly when only the sum of the bounds is over the budget" in:
    // 6D surface, level 3: the bound is 3,317,760 faces (6.6M triangles, within the 16M budget),
    // the sponge has 2,426,880 faces. Three of them: 19.9M triangles by their bounds, over the
    // budget, and 14.6M in fact, which fits.
    val spec = ndSponge(NdSpongeVariant.Surface, 6, 3f)
    val bound = NdSponge.upperBoundFaces(NdSpongeVariant.Surface, 6, 3, 1)
    bound * TrianglesPerFace should be <= Budget
    SceneCost.triangles(spec) shouldBe TrianglesPerFace * bound
    val scene = List.fill(3)(spec)
    scene.map(SceneCost.triangles).sum should be > Budget
    SceneCost.estimate(scene, 1, SceneCost.CausticsLoad.None).triangles shouldBe
      3 * TrianglesPerFace * 2426880L
    fits(scene) shouldBe true

  it should "refuse a level whose exact count is over the budget, and name the level that fits" in:
    // 6D volume, level 2, has 5,898,240 faces: one fits, two do not (23.6M triangles).
    val spec = ndSponge(NdSpongeVariant.Volume, 6, 2f)
    SceneCost.triangles(spec) shouldBe TrianglesPerFace * 5898240L
    fits(List(spec)) shouldBe true
    fits(List(spec, spec)) shouldBe false
    val fit = SceneCost.fittingLevel(List(spec, spec))
    fit.map(_.objectType) shouldBe Some("n-sponge-volume")
    fit.map(_.level) shouldBe Some(1)
    SceneCost.checkTriangles(List(spec, spec)).left.getOrElse("") should
      include("the n-sponge-volume fits at level 1")

  it should "refuse a level past the bound's headroom without building it" in:
    // 7D volume, level 2: 223M faces by the bound, far over 4x the budget; a build would not fit.
    val spec = ndSponge(NdSpongeVariant.Volume, 7, 2f)
    val bound = NdSponge.upperBoundFaces(NdSpongeVariant.Volume, 7, 2, 1)
    ResourceLimits.ndSpongeBoundAdmits(bound) shouldBe false
    SceneCost.triangles(spec) should be > Budget
    fits(List(spec)) shouldBe false
    SceneCost.fittingLevel(List(spec)).map(_.level) shouldBe Some(1)

  it should "count a fractional level as both whole levels, the upper one too" in:
    val lower = ndSponge(NdSpongeVariant.Surface, 5, 1f)
    val upper = ndSponge(NdSpongeVariant.Surface, 5, 2f)
    SceneCost.triangles(ndSponge(NdSpongeVariant.Surface, 5, 1.4f)) shouldBe
      SceneCost.triangles(lower) + SceneCost.triangles(upper)

  it should "not build a mesh to cost a sponge" in:
    val before = NdSpongeScene.retainedFaces
    SceneCost.estimate(
      List.fill(3)(ndSponge(NdSpongeVariant.Surface, 6, 3f)), 1, SceneCost.CausticsLoad.None
    )
    SceneCost.estimate(
      List(ndSponge(NdSpongeVariant.Volume, 6, 2f)), 1, SceneCost.CausticsLoad.None
    )
    NdSpongeScene.retainedFaces shouldBe before

  it should "fail with a configuration error for a spec without a dimension" in:
    val spec = ndSponge(NdSpongeVariant.Surface, 5, 1f).copy(dimension = None)
    a[ConfigurationException] should be thrownBy SceneCost.triangles(spec)

  // The volume is counted exactly however loose its bound is: 5D volume at level 2 has 1,003,520
  // faces by the bound and 450,560 in fact. Nine of them are 8.1M triangles.
  it should "count volume sponges exactly, however loose their bound" in:
    val spec = ndSponge(NdSpongeVariant.Volume, 5, 2f)
    val scene = List.fill(9)(spec)
    val estimate = SceneCost.estimate(scene, 1, SceneCost.CausticsLoad.None)
    estimate.triangles shouldBe 9 * TrianglesPerFace * 450560L
    estimate.fittingLevel shouldBe None
    fits(scene) shouldBe true
    SceneCost.checkTriangles(scene) shouldBe Right(())

  it should "still refuse a scene whose exact counts are over the budget" in:
    // 19 x 901,120 = 17.1M triangles in fact; without the 18 others no level of the last fits
    val scene = List.fill(19)(ndSponge(NdSpongeVariant.Volume, 5, 2f))
    val estimate = SceneCost.estimate(scene, 1, SceneCost.CausticsLoad.None)
    estimate.triangles shouldBe 19 * TrianglesPerFace * 450560L
    SceneCost.check(estimate).left.getOrElse("") should include("triangles in total")

  // k > 1 keeps more sub-cubes: the faces differ from k = 1 at the same level (menger#166).
  it should "count a larger k exactly: 6D surface, level 3, k = 3" in:
    val spec = ndSponge(NdSpongeVariant.Surface, 6, 3f, k = 3)
    SceneCost.triangles(spec) shouldBe TrianglesPerFace * 3317760L
    NdSponge.faceCount(NdSpongeVariant.Surface, 6, 3, 3) should be >
      NdSponge.faceCount(NdSpongeVariant.Surface, 6, 3, 1)
    fits(List(spec)) shouldBe true

  it should "refuse three of them by their exact count, and name the level that fits" in:
    val spec = ndSponge(NdSpongeVariant.Surface, 6, 3f, k = 3)
    fits(List(spec, spec)) shouldBe true
    val scene = List(spec, spec, spec)
    SceneCost.estimate(scene, 1, SceneCost.CausticsLoad.None).triangles shouldBe
      3 * TrianglesPerFace * 3317760L
    fits(scene) shouldBe false
    SceneCost.fittingLevel(scene).map(_.level) shouldBe Some(2)

  it should "refuse a larger k whose bound is past the headroom, and name the level that fits" in:
    // 6D volume, level 2, k = 2: 59M faces by the bound, which a k = 1 sponge would not have
    val spec = ndSponge(NdSpongeVariant.Volume, 6, 2f, k = 2)
    ResourceLimits.ndSpongeBoundAdmits(
      NdSponge.upperBoundFaces(NdSpongeVariant.Volume, 6, 2, 2)
    ) shouldBe false
    fits(List(spec)) shouldBe false
    fits(List(ndSponge(NdSpongeVariant.Volume, 6, 2f))) shouldBe true
    SceneCost.fittingLevel(List(spec)).map(_.level) shouldBe Some(1)

  "The n-sponge level limit" should "admit the highest level by its bound and no higher one" in:
    for
      variant <- NdSpongeVariant.values
      n <- 4 to 8
    do
      val max = ResourceLimits.ndSpongeLevel(variant, n).max
      def boundAt(level: Int) = NdSponge.upperBoundFaces(variant, n, level, 1)
      ResourceLimits.ndSpongeBoundAdmits(boundAt(max)) shouldBe true
      val next = max + 1
      if NdSponge.isBuildable(variant, n, next, 1) then
        ResourceLimits.ndSpongeBoundAdmits(boundAt(next)) shouldBe false
