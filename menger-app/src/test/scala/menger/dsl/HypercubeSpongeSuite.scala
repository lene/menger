package menger.dsl

import menger.objects.higher_d.NdSpongeVariant
import menger.spec.ProjectionSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The DSL's nD Menger sponge (menger#166): dimension 4..8, k in 1..N-1, a level within the
  * dimension's triangle budget, a view of at least N dimensions, and the spec the renderer gets. */
class HypercubeSpongeSuite extends AnyFlatSpec with Matchers:

  private val Volume = TesseractSpongeType.VolumeRemoving
  private val Surface = TesseractSpongeType.SurfaceSubdividing

  "HypercubeSponge" should "be constructible from its dimension, type and level" in:
    val sponge = HypercubeSponge(5, Surface, 2f)
    (sponge.dimension, sponge.level, sponge.maxCentreCoordinates) shouldBe ((5, 2f, 1))
    sponge.pos shouldBe Vec3.Zero
    sponge.size shouldBe 1.0f
    sponge.projection shouldBe None

  it should "become the spec of the matching object type, with its dimension, k and default view" in:
    val volume = HypercubeSponge(6, Volume, 1f, maxCentreCoordinates = 4).toObjectSpec
    (volume.objectType, volume.dimension, volume.maxCentreCoordinates, volume.level) shouldBe
      (("n-sponge-volume", Some(6), Some(4), Some(1f)))
    volume.projectionN shouldBe Some(ProjectionSpec(6))
    val surface = HypercubeSponge(5, Surface, 2f).toObjectSpec
    (surface.objectType, surface.maxCentreCoordinates) shouldBe (("n-sponge-surface", Some(1)))

  it should "carry its own view, scale, size and material into the spec" in:
    val view = ProjectionSpec(7)
    val spec = HypercubeSponge(
      5, Surface, 1f, projection = Some(view), size = 2f, material = Some(Material.Glass),
      scale = Some(Scale.identity[5].withLast(0.5f)), pos = Vec3(1f, 2f, 3f)
    ).toObjectSpec
    spec.projectionN shouldBe Some(view)
    spec.scale shouldBe Some(IndexedSeq(1f, 1f, 1f, 1f, 0.5f))
    (spec.size, spec.x, spec.y, spec.z) shouldBe ((2f, 1f, 2f, 3f))
    spec.material should not be empty

  it should "accept every dimension from 4 to 8 at level 0" in:
    (4 to 8).foreach(n => HypercubeSponge(n, Volume, 0f).toObjectSpec.dimension shouldBe Some(n))

  it should "reject a dimension outside 4..8, naming the range" in:
    Seq(-1, 0, 3, 9).foreach { n =>
      val refusal = the[IllegalArgumentException] thrownBy HypercubeSponge(n, Volume, 0f)
      refusal.getMessage should (include("HypercubeSponge.dimension") and include("4-8"))
    }

  it should "reject k of 0 or N, naming 1..N-1" in:
    Seq((5, 0), (5, 5), (6, 6), (4, 4), (8, 0)).foreach { (n, k) =>
      val refusal = the[IllegalArgumentException] thrownBy
        HypercubeSponge(n, Volume, 0f, maxCentreCoordinates = k)
      refusal.getMessage should include(s"1..${n - 1}")
    }

  it should "accept k from 1 to N-1" in:
    (4 to 8).foreach { n =>
      (1 until n).foreach(k =>
        HypercubeSponge(n, Surface, 0f, maxCentreCoordinates = k).toObjectSpec
          .maxCentreCoordinates shouldBe Some(k)
      )
    }

  it should "accept the highest level of each dimension and refuse one more, naming the maximum" in:
    for
      (variant, spongeType) <- Seq(NdSpongeVariant.Volume -> Volume, NdSpongeVariant.Surface -> Surface)
      n <- 4 to 8
    do
      val max = ResourceLimits.ndSpongeLevel(variant, n).max
      HypercubeSponge(n, spongeType, max.toFloat).level shouldBe max.toFloat
      val refusal = the[IllegalArgumentException] thrownBy
        HypercubeSponge(n, spongeType, max + 1f)
      refusal.getMessage should (include(s"maximum $max") and include(s"dimension $n"))

  it should "accept a fractional level up to the maximum" in:
    HypercubeSponge(5, Volume, 1.4f).level shouldBe 1.4f
    an[IllegalArgumentException] should be thrownBy HypercubeSponge(5, Volume, 2.1f)

  it should "reject a negative level" in:
    an[IllegalArgumentException] should be thrownBy HypercubeSponge(5, Volume, -0.5f)

  it should "reject a projection of fewer dimensions than the sponge, naming both" in:
    val refusal = the[IllegalArgumentException] thrownBy
      HypercubeSponge(6, Volume, 1f, projection = Some(ProjectionSpec(5)))
    refusal.getMessage should (include("6-dimensional") and include("dimension = 5"))

  it should "accept a projection of the sponge's dimension or more" in:
    HypercubeSponge(5, Volume, 1f, projection = Some(ProjectionSpec(5))).projection should
      not be empty
    HypercubeSponge(5, Volume, 1f, projection = Some(ProjectionSpec(8))).projection should
      not be empty

  it should "reject a Scale of another dimension, naming both" in:
    val refusal = the[IllegalArgumentException] thrownBy
      HypercubeSponge(6, Volume, 1f, scale = Some(Scale[5](1f, 1f, 1f, 1f, 0.5f)))
    refusal.getMessage should (include("6") and include("5"))

  it should "carry no scale for an identity Scale" in:
    HypercubeSponge(5, Volume, 1f, scale = Some(Scale.identity[5])).toObjectSpec.scale shouldBe None

  it should "validate positive size and non-negative ior" in:
    an[IllegalArgumentException] should be thrownBy HypercubeSponge(5, Volume, 1f, size = 0f)
    an[IllegalArgumentException] should be thrownBy HypercubeSponge(5, Volume, 1f, ior = -1f)

  "The level limits" should "be the upper limits of each dimension at today's 16M budget" in:
    // volume: 80 * 112^L faces at N = 5; each level multiplies by 112, so the bound of two
    // triangles per face stays within four budgets for L = 2 and not for L = 3
    ResourceLimits.ndSpongeLevel(NdSpongeVariant.Volume, 5).max shouldBe 2
    ResourceLimits.ndSpongeLevel(NdSpongeVariant.Surface, 5).max shouldBe 3
    ResourceLimits.ndSpongeLevel(NdSpongeVariant.Volume, 8).max shouldBe 1
    ResourceLimits.ndSpongeLevel(NdSpongeVariant.Surface, 8).max shouldBe 2

  it should "never allow more levels in a higher dimension" in:
    for variant <- NdSpongeVariant.values do
      val maxima = (4 to 8).map(ResourceLimits.ndSpongeLevel(variant, _).max)
      maxima shouldBe maxima.sorted.reverse

  it should "be bounded in the manifest by the loosest dimension of each variant" in:
    val limits = FieldConstraints.bySubtype(("HypercubeSponge", "level"))
    limits.field shouldBe "spongeType"
    limits.values("VolumeRemoving").max shouldBe Some(
      (4 to 8).map(ResourceLimits.ndSpongeLevel(NdSpongeVariant.Volume, _).max).max.toDouble
    )
    limits.values("SurfaceSubdividing").max shouldBe Some(
      (4 to 8).map(ResourceLimits.ndSpongeLevel(NdSpongeVariant.Surface, _).max).max.toDouble
    )
    FieldConstraints.fields(("HypercubeSponge", "dimension")).admits(4d) shouldBe true
    FieldConstraints.fields(("HypercubeSponge", "dimension")).admits(9d) shouldBe false
    FieldConstraints.fields(("HypercubeSponge", "maxCentreCoordinates")).admits(7d) shouldBe true
    FieldConstraints.fields(("HypercubeSponge", "maxCentreCoordinates")).admits(8d) shouldBe false
