package io.github.lene.optix

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Which renders a frame with level-mixed objects needs, and how they are combined. */
class LevelMixSuite extends AnyFlatSpec with Matchers:

  private def weights(worlds: Vector[(Vector[Boolean], Double)]) = worlds.map(_._2)

  "LevelMix.worlds" should "render one object's lower and upper level, weighted 1 - f and f" in:
    LevelMix.worlds(Vector(0.25f)) shouldBe Vector((Vector(false), 0.75), (Vector(true), 0.25))

  it should "render every combination of two objects' levels, weighted by the product" in:
    val worlds = LevelMix.worlds(Vector(0.25f, 0.75f))
    worlds.map(_._1) should contain theSameElementsAs Seq(
      Vector(false, false), Vector(true, false), Vector(false, true), Vector(true, true)
    )
    worlds.toMap.apply(Vector(true, false)) shouldBe (0.25 * 0.25 +- 1e-9)
    worlds.toMap.apply(Vector(false, true)) shouldBe (0.75 * 0.75 +- 1e-9)
    weights(worlds).sum shouldBe (1.0 +- 1e-9)

  it should "render 2^k combinations up to the exact limit" in:
    LevelMix.worlds(Vector.fill(LevelMix.MaxExactMixes)(0.5f)).size shouldBe
      (1 << LevelMix.MaxExactMixes)

  it should "fall back to k + 1 renders beyond it: all lower, then each object alone upper" in:
    val k = LevelMix.MaxExactMixes + 1
    val worlds = LevelMix.worlds(Vector.fill(k)(0.4f))
    worlds.size shouldBe k + 1
    worlds.head shouldBe ((Vector.fill(k)(false), 1.0 - k * 0.4f.toDouble))
    worlds.tail.map(_._1.count(identity)) shouldBe Vector.fill(k)(1)
    weights(worlds).sum shouldBe (1.0 +- 1e-6)

  "LevelMix.mix" should "weight and round each channel" in:
    val mixed = LevelMix.mix(Seq((Array[Byte](0, 100, -1), 0.75), (Array[Byte](100, 0, -1), 0.25)))
    mixed.map(_ & 0xff).toSeq shouldBe Seq(25, 75, 255)

  it should "clamp to 0..255 where the first-order sum weights a render negatively" in:
    val mixed = LevelMix.mix(Seq((Array[Byte](10, -56), -0.5), (Array[Byte](10, -56), 2.0)))
    mixed.map(_ & 0xff).toSeq shouldBe Seq(15, 255)

  "LevelMix" should "need a fraction strictly between 0 and 1" in:
    val (one, two) = (InstanceId(1), InstanceId(2))
    an[IllegalArgumentException] should be thrownBy LevelMix(Vector(one), Vector(two), 0f)
    an[IllegalArgumentException] should be thrownBy LevelMix(Vector(one), Vector(two), 1f)
    LevelMix(Vector(one), Vector(two), 0.5f).contains(two) shouldBe true
