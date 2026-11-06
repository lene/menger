package menger.objects.higher_d

import menger.PropertyTestGenerators.angleGen
import menger.PropertyTestGenerators.dimensions
import menger.PropertyTestGenerators.planeGen
import menger.PropertyTestGenerators.vectorGen
import menger.common.Vector
import menger.objects.Matrix
import menger.objects.higher_d.CustomMatchers.closeTo
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class NdRotationPropertySuite extends AnyFlatSpec with Matchers with ScalaCheckPropertyChecks:

  private val tolerance = 1e-3f
  private val fullTurn = 360f

  private def bits(values: Iterable[Float]): Seq[Int] =
    values.map(java.lang.Float.floatToRawIntBits).toSeq

  "A plane rotation in dimension 3..8" should "be orthogonal: R * R^T = I" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      forAll(planeGen(d), angleGen) { (plane, angle) =>
        val m = Rotation.planeMatrix[d.N](plane, angle)
        (m * m.transpose) should closeTo(Matrix.identity[d.N], tolerance)
        (m.transpose * m) should closeTo(Matrix.identity[d.N], tolerance)
      }
    }

  it should "preserve norm and distance" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      forAll(planeGen(d), angleGen, vectorGen(d), vectorGen(d)) { (plane, angle, p, q) =>
        val rotation = Rotation.inPlane[d.N](plane, angle)
        rotation(p).len shouldBe p.len +- tolerance
        rotation(p).dst(rotation(q)) shouldBe p.dst(q) +- tolerance
      }
    }

  it should "change only the coordinates of its plane" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      forAll(planeGen(d), angleGen, vectorGen(d)) { (plane, angle, p) =>
        val rotated = Rotation.inPlane[d.N](plane, angle)(p)
        (0 until d.value).filterNot(plane.indices.contains).foreach { k =>
          rotated(k) shouldBe p(k)
        }
      }
    }

  it should "be the identity after 360 degrees" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      forAll(planeGen(d), vectorGen(d)) { (plane, p) =>
        Rotation.inPlane[d.N](plane, fullTurn)(p) should closeTo(p)
      }
    }

  it should "turn by theta in (i, j) exactly as by -theta in (j, i)" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      forAll(planeGen(d), angleGen, vectorGen(d)) { (plane, angle, p) =>
        val swapped = Plane(plane.j, plane.i, d.value)
        Rotation.inPlane[d.N](swapped, -angle)(p) should closeTo(
          Rotation.inPlane[d.N](plane, angle)(p), tolerance
        )
      }
    }

  it should "turn the second axis of its plane onto the first by 90 degrees" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      forAll(planeGen(d)) { plane =>
        val unitJ = Vector.unit[d.N](plane.j)
        Rotation.inPlane[d.N](plane, 90f)(unitJ) should closeTo(Vector.unit[d.N](plane.i))
      }
    }

  "Rotations in written order" should "turn the first plane first" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      forAll(planeGen(d), angleGen, planeGen(d), angleGen, vectorGen(d)) {
        (p1, a1, p2, a2, point) =>
          val composed = Rotation.inWrittenOrder[d.N](Seq((p1, a1), (p2, a2)))
          val stepwise = Rotation.inPlane[d.N](p2, a2)(Rotation.inPlane[d.N](p1, a1)(point))
          composed(point) should closeTo(stepwise, tolerance)
      }
    }

  it should "differ from the reversed order for planes sharing one axis" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      val (first, second) = (Plane(0, 1, d.value), Plane(1, 2, d.value))
      val point = Vector.unit[d.N](1)
      val written = Rotation.inWrittenOrder[d.N](Seq((first, 90f), (second, 90f)))(point)
      val reversed = Rotation.inWrittenOrder[d.N](Seq((second, 90f), (first, 90f)))(point)
      written should not be closeTo(reversed, tolerance)
    }

  it should "be the identity for an empty list and the plane rotation for a single turn" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      val plane = Plane(0, 1, d.value)
      Rotation.inWrittenOrder[d.N](Seq.empty).isZero shouldBe true
      bits(Rotation.inWrittenOrder[d.N](Seq((plane, 33f))).transformationMatrix.m) shouldBe
        bits(Rotation.planeMatrix[d.N](plane, 33f).m)
    }

  "A rotation about a pivot" should "keep the pivot fixed and rotate p - pivot about it" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      forAll(planeGen(d), angleGen, planeGen(d), angleGen, vectorGen(d), vectorGen(d)) {
        (p1, a1, p2, a2, pivot, point) =>
          val single = Rotation.inPlane[d.N](p1, a1, pivot)
          val written = Rotation.inWrittenOrder[d.N](Seq((p1, a1), (p2, a2)), pivot)
          val aboutOrigin = Rotation.inPlane[d.N](p1, a1)
          val writtenAboutOrigin = Rotation.inWrittenOrder[d.N](Seq((p1, a1), (p2, a2)))
          single(pivot) should closeTo(pivot, tolerance)
          written(pivot) should closeTo(pivot, tolerance)
          single(point) should closeTo(aboutOrigin(point - pivot) + pivot, tolerance)
          written(point) should closeTo(writtenAboutOrigin(point - pivot) + pivot, tolerance)
      }
    }

  "A 4D-only Plane operation" should "reject a plane of another dimension, naming 4D planes" in:
    val neg = the[IllegalArgumentException] thrownBy Plane(0, 1, 5).neg
    neg.getMessage should (include("4D planes") and include("dimension 5"))
    val units = the[IllegalArgumentException] thrownBy Plane(0, 1, 5).units
    units.getMessage should (include("4D planes") and include("dimension 5"))

  "A plane rotation in dimension 4" should "be bit-identical to the legacy factories" in:
    val planes = Seq(Plane.xy, Plane.xz, Plane.xw, Plane.yz, Plane.yw, Plane.zw)
    forAll(angleGen) { angle =>
      planes.foreach { plane =>
        bits(Rotation.inPlane[4](plane, angle).transformationMatrix.m) shouldBe
          bits(Rotation(plane, angle).transformationMatrix.m)
      }
    }

  it should "compose XW, YW, ZW in written order like the legacy factory, within float noise" in:
    forAll(angleGen, angleGen, angleGen) { (xw, yw, zw) =>
      val written = Rotation.inWrittenOrder[4](Seq((Plane.zw, zw), (Plane.yw, yw), (Plane.xw, xw)))
      written.transformationMatrix should closeTo(
        Rotation(xw, yw, zw).transformationMatrix, 1e-5f
      )
    }

  "Rotating a point with the last two axes" should "move its perspective projection but not its orthographic one" in:
    val (eye, screen) = (3f, 1f)
    dimensions.filter(_.value >= 5).foreach { d =>
      given ValueOf[d.N] = d.valueOf
      val lastTwo = Plane(d.value - 2, d.value - 1, d.value)
      val extra = Vector.fromSeq[d.N](Seq.tabulate(d.value) {
        case k if k == d.value - 1 => 1f
        case k if k < 3 => (k + 1) * 0.5f
        case _ => 0f
      })
      val rotated = Rotation.inPlane[d.N](lastTwo, 90f)(extra)
      def orthographic(v: Vector[d.N]) = Vector[3](v(0), v(1), v(2))
      orthographic(rotated) should closeTo(orthographic(extra))
      val chain = PerspectiveChain.uniform[d.N](eye, screen)
      chain(rotated).dst(chain(extra)) should be > 0.01f
    }

  "Rejecting an invalid plane or dimension" should "name the plane and the dimension" in:
    val sameAxis = the[IllegalArgumentException] thrownBy Plane(2, 2, 5)
    sameAxis.getMessage should (include("must differ") and include("dimension 5"))
    val outOfRange = the[IllegalArgumentException] thrownBy Plane(0, 5, 5)
    outOfRange.getMessage should (include("0 and 4") and include("dimension 5") and include("5"))
    val tooBig = the[IllegalArgumentException] thrownBy Plane(0, 1, 9)
    tooBig.getMessage should (include("9") and include("(plane (0, 1))"))
    val tooSmall = the[IllegalArgumentException] thrownBy Plane(0, 1, 2)
    tooSmall.getMessage should (include("is 2") and include("(plane (0, 1))"))

  it should "reject a plane of another dimension than the rotation's" in:
    val mismatch = the[IllegalArgumentException] thrownBy Rotation.inPlane[5](Plane.xw, 10f)
    mismatch.getMessage should (include("xw") and include("dimension 4") and include("dimension 5"))

  it should "reject a rotation dimension outside 3..8" in:
    val unsupported = the[IllegalArgumentException] thrownBy
      Rotation.planeMatrix[9](Plane.xy, 10f)
    unsupported.getMessage should (include("9") and include("xy"))

  "Plane letters" should "name the axes x y z w v u t s" in:
    Plane(3, 4, 5).toString shouldBe "wv"
    Plane(4, 5, 6).toString shouldBe "vu"
    Plane(5, 6, 7).toString shouldBe "ut"
    Plane(6, 7, 8).toString shouldBe "ts"
    Plane(0, 7, 8).toString shouldBe "xs"

  "Dimension.of" should "give the ValueOf matching its value for 3..8 and reject others" in:
    dimensions.map(_.value) shouldBe (3 to 8)
    dimensions.foreach(d => d.valueOf.value shouldBe d.value)
    an[IllegalArgumentException] should be thrownBy Dimension.of(2)
    an[IllegalArgumentException] should be thrownBy Dimension.of(9)

  "Matrix.transpose" should "swap rows and columns and keep its dimension" in:
    val m = Matrix[3](Array[Float](1, 2, 3, 4, 5, 6, 7, 8, 9))
    bits(m.transpose.m) shouldBe bits(Seq[Float](1, 4, 7, 2, 5, 8, 3, 6, 9))
    bits(m.transpose.transpose.m) shouldBe bits(m.m)
