package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CausticsSuite extends AnyFlatSpec with Matchers:

  "Caustics" should "have correct defaults" in:
    val caustics = Caustics()
    caustics.enabled shouldBe true
    caustics.photonsPerIteration shouldBe 100000
    caustics.iterations shouldBe 10
    caustics.initialRadius shouldBe None
    caustics.alpha shouldBe 0.7f

  it should "map a None radius to the auto-derive sentinel" in:
    Caustics().toCausticsConfig.initialRadius shouldBe menger.common.CausticsConfig.AutoRadius

  it should "be constructible with custom parameters" in:
    val caustics = Caustics(
      enabled = true,
      photonsPerIteration = 200000,
      iterations = 15,
      initialRadius = Some(0.2f),
      alpha = 0.8f
    )
    caustics.photonsPerIteration shouldBe 200000
    caustics.iterations shouldBe 15
    caustics.initialRadius shouldBe Some(0.2f)
    caustics.alpha shouldBe 0.8f

  it should "validate photonsPerIteration range" in:
    an[IllegalArgumentException] should be thrownBy Caustics(photonsPerIteration = 0)
    an[IllegalArgumentException] should be thrownBy Caustics(photonsPerIteration = 10000001)

  it should "validate iterations range" in:
    an[IllegalArgumentException] should be thrownBy Caustics(iterations = 0)
    an[IllegalArgumentException] should be thrownBy Caustics(iterations = 1001)

  it should "validate initialRadius range" in:
    an[IllegalArgumentException] should be thrownBy Caustics(initialRadius = Some(-0.1f))
    an[IllegalArgumentException] should be thrownBy Caustics(initialRadius = Some(10.1f))

  it should "accept a None radius (auto-derive) without throwing" in:
    noException should be thrownBy Caustics(initialRadius = None)

  it should "validate alpha range" in:
    an[IllegalArgumentException] should be thrownBy Caustics(alpha = 0f)
    an[IllegalArgumentException] should be thrownBy Caustics(alpha = 1f)

  "Caustics.Disabled" should "have correct configuration" in:
    val caustics = Caustics.Disabled
    caustics.enabled shouldBe false

  "Caustics.Default" should "have correct configuration" in:
    val caustics = Caustics.Default
    caustics.enabled shouldBe true
    caustics.photonsPerIteration shouldBe 100000

  "Caustics.HighQuality" should "have correct configuration" in:
    val caustics = Caustics.HighQuality
    caustics.enabled shouldBe true
    caustics.photonsPerIteration shouldBe 500000
    caustics.iterations shouldBe 20
    caustics.initialRadius shouldBe Some(1.0f)
    caustics.alpha shouldBe 0.8f

  "Caustics.toCausticsConfig" should "convert to OptiX CausticsConfig correctly" in:
    val dsl = Caustics(
      enabled = true,
      photonsPerIteration = 150000,
      iterations = 12,
      initialRadius = Some(0.15f),
      alpha = 0.75f
    )
    val config = dsl.toCausticsConfig

    config.enabled shouldBe true
    config.photonsPerIteration shouldBe 150000
    config.iterations shouldBe 12
    config.initialRadius shouldBe 0.15f
    config.alpha shouldBe 0.75f

  it should "convert Disabled correctly" in:
    val config = Caustics.Disabled.toCausticsConfig
    config.enabled shouldBe false
