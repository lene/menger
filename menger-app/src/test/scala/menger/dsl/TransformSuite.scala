package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TransformSuite extends AnyFlatSpec with Matchers:

  "Transform.Identity" should "have zero translation" in:
    Transform.Identity.translation shouldBe Vec3.Zero

  it should "have zero rotation" in:
    Transform.Identity.rotation shouldBe Vec3.Zero

  it should "have scale 1.0" in:
    Transform.Identity.scale shouldBe 1.0f

  "Transform.at" should "create a translation-only transform" in:
    val t = Transform.at(Vec3(3f, 0f, 0f))
    t.translation shouldBe Vec3(3f, 0f, 0f)
    t.rotation shouldBe Vec3.Zero
    t.scale shouldBe 1.0f

  "Transform.scaled" should "create a scale-only transform" in:
    val t = Transform.scaled(2f)
    t.translation shouldBe Vec3.Zero
    t.rotation shouldBe Vec3.Zero
    t.scale shouldBe 2.0f

  "Transform.accumulate" should "compose two identity transforms to identity" in:
    val result = Transform.accumulate(Transform.Identity, Transform.Identity)
    result shouldBe Transform.Identity

  it should "add translations (parent at origin, no parent scale effect)" in:
    val parent = Transform.at(Vec3(1f, 0f, 0f))
    val child  = Transform(translation = Vec3(2f, 0f, 0f))
    val result = Transform.accumulate(parent, child)
    result.translation.x shouldBe (1f + 1f * 2f) +- 1e-5f
    result.translation.y shouldBe 0f +- 1e-5f
    result.translation.z shouldBe 0f +- 1e-5f

  it should "scale child translation by parent scale" in:
    val parent = Transform(scale = 2f)
    val child  = Transform(translation = Vec3(1f, 0f, 0f))
    val result = Transform.accumulate(parent, child)
    result.translation.x shouldBe (0f + 2f * 1f) +- 1e-5f

  it should "multiply scales" in:
    val parent = Transform.scaled(2f)
    val child  = Transform.scaled(3f)
    val result = Transform.accumulate(parent, child)
    result.scale shouldBe (2f * 3f) +- 1e-5f

  it should "add rotations (Euler approximation)" in:
    val parent = Transform(rotation = Vec3(0f, 0.5f, 0f))
    val child  = Transform(rotation = Vec3(0f, 0.5f, 0f))
    val result = Transform.accumulate(parent, child)
    result.rotation.y shouldBe 1.0f +- 1e-5f

  it should "keep identity transform when composing with identity" in:
    val t = Transform(translation = Vec3(5f, 3f, 1f), rotation = Vec3(0.1f, 0.2f, 0.3f), scale = 2f)
    Transform.accumulate(t, Transform.Identity) shouldBe t

  it should "chain multiple transforms correctly" in:
    val a = Transform(translation = Vec3(1f, 0f, 0f), scale = 2f)
    val b = Transform(translation = Vec3(1f, 0f, 0f), scale = 3f)
    val c = Transform(translation = Vec3(1f, 0f, 0f))
    val ab = Transform.accumulate(a, b)
    val abc = Transform.accumulate(ab, c)
    // a: t=(1,0,0), s=2
    // b local: t=(1,0,0), s=3
    // ab: t=(1+2*1, 0, 0)=(3,0,0), s=6
    // c local: t=(1,0,0), s=1
    // abc: t=(3+6*1,0,0)=(9,0,0), s=6
    abc.translation.x shouldBe 9f +- 1e-4f
    abc.scale shouldBe 6f +- 1e-4f
