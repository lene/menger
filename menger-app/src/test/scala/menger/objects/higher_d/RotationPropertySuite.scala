package menger.objects.higher_d

import menger.PropertyTestGenerators.angleGen
import menger.PropertyTestGenerators.smallVector4Gen
import menger.objects.Matrix
import menger.objects.higher_d.CustomMatchers._
import org.scalacheck.Gen
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class RotationPropertySuite extends AnyFlatSpec with Matchers with ScalaCheckPropertyChecks:

  val planeGen: Gen[Plane] = Gen.oneOf(Plane.xy, Plane.xz, Plane.xw, Plane.yz, Plane.yw, Plane.zw)

  "Rotation by 360 degrees" should "return to original position" in:
    forAll(planeGen, smallVector4Gen) { (plane, v) =>
      val rotation = Rotation(plane, 360f)
      rotation(v) should epsilonEqual(v)
    }

  "Rotation by 0 degrees" should "be identity" in:
    forAll(planeGen, smallVector4Gen) { (plane, v) =>
      val rotation = Rotation(plane, 0f)
      rotation(v) should epsilonEqual(v)
    }

  "Rotation followed by inverse rotation" should "return to original" in:
    forAll(planeGen, angleGen, smallVector4Gen) { (plane, angle, v) =>
      val rotation = Rotation(plane, angle)
      val inverse = Rotation(plane, -angle)
      (inverse * rotation)(v) should epsilonEqual(v)
    }

  "Rotation" should "preserve vector length" in:
    forAll(planeGen, angleGen, smallVector4Gen) { (plane, angle, v) =>
      val rotation = Rotation(plane, angle)
      rotation(v).len shouldBe v.len +- 0.01f
    }

  it should "preserve distance between points" in:
    forAll(planeGen, angleGen, smallVector4Gen, smallVector4Gen) { (plane, angle, v1, v2) =>
      val rotation = Rotation(plane, angle)
      val originalDist = v1.dst(v2)
      val rotatedDist = rotation(v1).dst(rotation(v2))
      rotatedDist shouldBe originalDist +- 0.01f
    }

  "Two rotations of 180 degrees" should "equal 360 degrees (identity)" in:
    forAll(planeGen, smallVector4Gen) { (plane, v) =>
      val rotation180 = Rotation(plane, 180f)
      val combined = rotation180 * rotation180
      combined(v) should epsilonEqual(v)
    }

  "Rotation by 90 degrees four times" should "return to original" in:
    forAll(planeGen, smallVector4Gen) { (plane, v) =>
      val rotation90 = Rotation(plane, 90f)
      val combined = rotation90 * rotation90 * rotation90 * rotation90
      combined(v) should epsilonEqual(v)
    }

  "Rotation composition" should "be associative" in:
    forAll(planeGen, angleGen, angleGen, angleGen, smallVector4Gen) { (plane, a1, a2, a3, v) =>
      val r1 = Rotation(plane, a1)
      val r2 = Rotation(plane, a2)
      val r3 = Rotation(plane, a3)
      ((r1 * r2) * r3)(v) should epsilonEqual((r1 * (r2 * r3))(v))
    }

  "Identity rotation" should "leave all vectors unchanged" in:
    forAll(smallVector4Gen) { v =>
      val identity = Rotation()
      identity(v) should epsilonEqual(v)
    }

  it should "have identity matrix" in:
    val identity = Rotation()
    identity.transformationMatrix should epsilonEqual(Matrix.identity[4])
