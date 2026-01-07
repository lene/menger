package menger.common

import org.scalacheck.Arbitrary
import org.scalacheck.Gen
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class VectorPropertySuite extends AnyFlatSpec with Matchers with ScalaCheckPropertyChecks:

  val reasonableFloat: Gen[Float] = Gen.choose(-1000f, 1000f)
  val smallFloat: Gen[Float] = Gen.choose(-10f, 10f)

  val vector4Gen: Gen[Vector[4]] = for
    x <- reasonableFloat
    y <- reasonableFloat
    z <- reasonableFloat
    w <- reasonableFloat
  yield Vector[4](x, y, z, w)

  val smallVector4Gen: Gen[Vector[4]] = for
    x <- smallFloat
    y <- smallFloat
    z <- smallFloat
    w <- smallFloat
  yield Vector[4](x, y, z, w)

  private def approxEqual(v1: Vector[4], v2: Vector[4], tolerance: Float = 0.001f): Boolean =
    v1.v.zip(v2.v).forall((a, b) => math.abs(a - b) < tolerance)

  "Vector addition" should "be commutative" in:
    forAll(vector4Gen, vector4Gen) { (a, b) =>
      (a + b) === (b + a) shouldBe true
    }

  it should "be associative for small vectors" in:
    forAll(smallVector4Gen, smallVector4Gen, smallVector4Gen) { (a, b, c) =>
      ((a + b) + c) === (a + (b + c)) shouldBe true
    }

  it should "have zero as identity element" in:
    forAll(vector4Gen) { v =>
      val zero = Vector.Zero[4]
      (v + zero) === v shouldBe true
    }

  it should "have additive inverse" in:
    forAll(vector4Gen) { v =>
      (v + (-v)) === Vector.Zero[4] shouldBe true
    }

  "Vector subtraction" should "be inverse of addition for small vectors" in:
    forAll(smallVector4Gen, smallVector4Gen) { (a, b) =>
      ((a + b) - b) === a shouldBe true
    }

  "Dot product" should "be commutative" in:
    forAll(vector4Gen, vector4Gen) { (a, b) =>
      (a * b) shouldBe (b * a) +- Const.epsilon
    }

  it should "distribute over addition for small vectors" in:
    forAll(smallVector4Gen, smallVector4Gen, smallVector4Gen) { (a, b, c) =>
      (a * (b + c)) shouldBe ((a * b) + (a * c)) +- 0.01f
    }

  it should "give zero with zero vector" in:
    forAll(vector4Gen) { v =>
      val zero = Vector.Zero[4]
      (v * zero) shouldBe 0f +- Const.epsilon
    }

  "Vector length" should "be non-negative" in:
    forAll(vector4Gen) { v =>
      v.len should be >= 0f
    }

  it should "be zero only for zero vector" in:
    val zero = Vector.Zero[4]
    zero.len shouldBe 0f +- Const.epsilon

  it should "satisfy len2 = len * len" in:
    forAll(smallVector4Gen) { v =>
      v.len2 shouldBe (v.len * v.len) +- 0.001f
    }

  it should "satisfy triangle inequality for small vectors" in:
    forAll(smallVector4Gen, smallVector4Gen) { (a, b) =>
      (a + b).len should be <= (a.len + b.len + Const.epsilon)
    }

  "Scalar multiplication" should "distribute over vector addition" in:
    forAll(smallVector4Gen, smallVector4Gen, smallFloat) { (a, b, s) =>
      approxEqual((a + b) * s, (a * s) + (b * s)) shouldBe true
    }

  it should "have 1 as identity" in:
    forAll(vector4Gen) { v =>
      (v * 1f) === v shouldBe true
    }

  it should "give zero when multiplied by 0" in:
    forAll(vector4Gen) { v =>
      (v * 0f) === Vector.Zero[4] shouldBe true
    }

  "Vector distance" should "be symmetric" in:
    forAll(vector4Gen, vector4Gen) { (a, b) =>
      a.dst(b) shouldBe b.dst(a) +- Const.epsilon
    }

  it should "be zero for same vector" in:
    forAll(vector4Gen) { v =>
      v.dst(v) shouldBe 0f +- Const.epsilon
    }

  it should "satisfy triangle inequality for small vectors" in:
    forAll(smallVector4Gen, smallVector4Gen, smallVector4Gen) { (a, b, c) =>
      a.dst(c) should be <= (a.dst(b) + b.dst(c) + Const.epsilon)
    }
