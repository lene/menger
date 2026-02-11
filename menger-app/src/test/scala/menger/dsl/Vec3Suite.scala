package menger.dsl

import scala.language.implicitConversions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Vec3Suite extends AnyFlatSpec with Matchers:

  "Vec3" should "be constructible with x, y, z components" in:
    val v = Vec3(1f, 2f, 3f)
    v.x shouldBe 1f
    v.y shouldBe 2f
    v.z shouldBe 3f

  it should "provide common constant vectors" in:
    Vec3.Zero shouldBe Vec3(0f, 0f, 0f)
    Vec3.UnitX shouldBe Vec3(1f, 0f, 0f)
    Vec3.UnitY shouldBe Vec3(0f, 1f, 0f)
    Vec3.UnitZ shouldBe Vec3(0f, 0f, 1f)

  it should "convert to libGDX Vector3" in:
    val v = Vec3(1f, 2f, 3f)
    val gdx = v.toGdxVector3
    gdx.x shouldBe 1f
    gdx.y shouldBe 2f
    gdx.z shouldBe 3f

  it should "convert to common Vector" in:
    val v = Vec3(1f, 2f, 3f)
    val common = v.toCommonVector
    common(0) shouldBe 1f
    common(1) shouldBe 2f
    common(2) shouldBe 3f

  "Vec3 tuple conversion" should "work with Float tuples" in:
    val v: Vec3 = (1.0f, 2.0f, 3.0f)
    v shouldBe Vec3(1f, 2f, 3f)

  it should "work with Int tuples" in:
    val v: Vec3 = (1, 2, 3)
    v shouldBe Vec3(1f, 2f, 3f)

  it should "work with Double tuples" in:
    val v: Vec3 = (1.0, 2.0, 3.0)
    v shouldBe Vec3(1f, 2f, 3f)

  it should "work in function parameters" in:
    def takesVec3(v: Vec3): Float = v.x + v.y + v.z
    takesVec3((1.0f, 2.0f, 3.0f)) shouldBe 6f
    takesVec3((1, 2, 3)) shouldBe 6f
