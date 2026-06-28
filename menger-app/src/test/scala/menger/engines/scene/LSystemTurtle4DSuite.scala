package menger.engines.scene

import menger.ObjectSpec
import menger.common.Vector
import menger.dsl.Vec3
import menger.objects.higher_d.Projection
import menger.objects.higher_d.Rotation
import org.scalatest.Inspectors.forAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LSystemTurtle4DSuite extends AnyFlatSpec with Matchers:

  private def dot(v1: Vector[4], v2: Vector[4]): Float =
    v1(0) * v2(0) + v1(1) * v2(1) + v1(2) * v2(2) + v1(3) * v2(3)
  private def magnitude(v: Vector[4]): Float =
    math.sqrt((0 to 3).map(i => v(i) * v(i)).sum.toDouble).toFloat

  "HilbertCurve4D preset" should "produce non-empty ObjectSpec list with curve type" in:
    val specs = LSystemTurtle4D.HilbertCurve4D.generate()
    specs should not be empty
    forAll(specs) { spec =>
      spec.objectType shouldBe "curve"
      spec.curveData should not be empty
    }

  it should "produce finite projected points" in:
    val specs = LSystemTurtle4D.HilbertCurve4D.generate()
    forAll(specs) { spec =>
      spec.curveData.foreach { cd =>
        forAll(cd.points) { p =>
          p.isFinite shouldBe true
          p.isNaN shouldBe false
          p.isInfinite shouldBe false
        }
      }
    }

  "Tree4D preset" should "produce non-empty ObjectSpec list" in:
    val specs = LSystemTurtle4D.Tree4D.generate()
    specs should not be empty
    forAll(specs) { spec =>
      spec.objectType shouldBe "curve"
    }

  "Frame orthonormality" should "be maintained after many rotations" in:
    val rng = new scala.util.Random(42L)
    val h = Vector.Y
    val l = Vector.X
    val u = Vector.Z
    val w = Vector.W
    val angle = 0.1f

    val planes = Seq(
      (1, 0), (1, 2), (0, 2), (1, 3)
    )

    def rotate(axes: (Vector[4], Vector[4], Vector[4], Vector[4]),
      plane: (Int, Int), ang: Float): (Vector[4], Vector[4], Vector[4], Vector[4]) =
      val (h2, l2, u2, w2) = axes
      val rh = LSystemTurtle4D.rotateVector4D(h2, plane._1, plane._2, ang)
      val rl = LSystemTurtle4D.rotateVector4D(l2, plane._1, plane._2, ang)
      val ru = LSystemTurtle4D.rotateVector4D(u2, plane._1, plane._2, ang)
      val rw = LSystemTurtle4D.rotateVector4D(w2, plane._1, plane._2, ang)
      (rh, rl, ru, rw)

    val finalAxes = (0 until 10000).foldLeft((h, l, u, w)) { (axes, _) =>
      val plane = planes(rng.nextInt(planes.length))
      val sign = if rng.nextBoolean() then 1f else -1f
      rotate(axes, plane, angle * sign)
    }

    val (fh, fl, fu, fw) = finalAxes

    magnitude(fh) shouldBe 1.0f +- 0.001f
    magnitude(fl) shouldBe 1.0f +- 0.001f
    magnitude(fu) shouldBe 1.0f +- 0.001f
    magnitude(fw) shouldBe 1.0f +- 0.001f

    math.abs(dot(fh, fl)).toDouble shouldBe 0.0 +- 0.001
    math.abs(dot(fh, fu)).toDouble shouldBe 0.0 +- 0.001
    math.abs(dot(fh, fw)).toDouble shouldBe 0.0 +- 0.001
    math.abs(dot(fl, fu)).toDouble shouldBe 0.0 +- 0.001
    math.abs(dot(fl, fw)).toDouble shouldBe 0.0 +- 0.001
    math.abs(dot(fu, fw)).toDouble shouldBe 0.0 +- 0.001

  "rotateVector4D" should "correctly rotate in the HL plane (yaw)" in:
    val v = Vector[4](1f, 0f, 0f, 0f)
    val piHalf = (math.Pi / 2.0).toFloat
    val rotated = LSystemTurtle4D.rotateVector4D(v, 1, 0, piHalf)
    rotated(0) shouldBe 0f +- 0.001f
    rotated(1) shouldBe -1f +- 0.001f
    rotated(2) shouldBe 0f +- 0.001f
    rotated(3) shouldBe 0f +- 0.001f

  it should "correctly rotate in the HW plane" in:
    val v = Vector[4](0f, 1f, 0f, 0f)
    val piHalf = (math.Pi / 2.0).toFloat
    val rotated = LSystemTurtle4D.rotateVector4D(v, 1, 3, piHalf)
    rotated(0) shouldBe 0f +- 0.001f
    rotated(1) shouldBe 0f +- 0.001f
    rotated(2) shouldBe 0f +- 0.001f
    rotated(3) shouldBe 1f +- 0.001f

  "4D rotation effect" should "produce different points with different rot-xw" in:
    val g = "F+F+F+F"
    val t1 = LSystemTurtle4D(g, 90f, 0.5f, rotXW = 0f)
    val t2 = LSystemTurtle4D(g, 90f, 0.5f, rotXW = 45f)
    val specs1 = t1.generate()
    val specs2 = t2.generate()

    def allPoints(specs: List[ObjectSpec]): scala.Vector[Float] =
      specs.flatMap(s => s.curveData.map(_.points).getOrElse(scala.Vector.empty)).toVector

    val pts1 = allPoints(specs1)
    val pts2 = allPoints(specs2)
    pts1 should not be pts2

  "Projection" should "produce finite 3D coordinates" in:
    val rot = Rotation(
      degreesXW = 15f, degreesYW = 10f, degreesZW = 0f, pivotPoint = Vector.Zero[4])
    val proj = Projection(eyeW = 3.0f, screenW = 1.5f)
    val v4 = Vector[4](1f, 2f, 3f, 0.5f)
    val v3 = LSystemTurtle4D.project4DTo3D(v4, rot, proj)
    v3.x.isFinite shouldBe true
    v3.y.isFinite shouldBe true
    v3.z.isFinite shouldBe true
    v3.x.isNaN shouldBe false
    v3.y.isNaN shouldBe false
    v3.z.isNaN shouldBe false

  it should "return non-zero 3D points for non-zero 4D input" in:
    val rot = Rotation(
      degreesXW = 0f, degreesYW = 0f, degreesZW = 0f, pivotPoint = Vector.Zero[4])
    val proj = Projection(eyeW = 3.0f, screenW = 1.5f)
    val v4 = Vector[4](1f, 1f, 1f, 0f)
    val v3 = LSystemTurtle4D.project4DTo3D(v4, rot, proj)
    val mag = math.sqrt(v3.x * v3.x + v3.y * v3.y + v3.z * v3.z).toDouble
    mag should be > 0.0

  "Push/pop" should "return to correct position after branch" in:
    val g = "FF[>+F]FF"
    val turtle = LSystemTurtle4D(g, 90f, 1.0f,
      rotXW = 0f, rotYW = 0f, rotZW = 0f, eyeW = 3.0f, screenW = 1.5f)
    val specs = turtle.generate()
    specs should not be empty

  "Unknown symbols" should "be skipped without error" in:
    val g = "F@OF+F"
    val turtle = LSystemTurtle4D(g, 90f, 1.0f)
    val specs = turtle.generate()
    specs should not be empty
    forAll(specs) { spec =>
      spec.objectType shouldBe "curve"
    }

  "Deterministic output" should "be identical for same seed" in:
    val g = "F+F-F-F+F"
    val t1 = LSystemTurtle4D(g, 90f, 0.5f, seed = 42L)
    val t2 = LSystemTurtle4D(g, 90f, 0.5f, seed = 42L)
    val specs1 = t1.generate()
    val specs2 = t2.generate()

    def flatPoints(specs: List[ObjectSpec]): scala.Vector[Float] =
      specs.flatMap(s =>
        s.curveData.map(_.points).getOrElse(scala.Vector.empty)
      ).toVector

    flatPoints(specs1) shouldBe flatPoints(specs2)

  "Width decay" should "produce decreasing widths" in:
    val g = "F!F!F!F"
    val turtle = LSystemTurtle4D(g, 0f, 1.0f, initialWidth = 1.0f, widthDecay = 0.5f)
    val specs = turtle.generate()
    specs should not be empty
    specs.head.curveData.foreach { cd =>
      cd.widths.size shouldBe 4
      cd.widths(0) shouldBe 1.0f
      cd.widths(1) shouldBe 0.5f +- 0.001f
      cd.widths(2) shouldBe 0.25f +- 0.001f
      cd.widths(3) shouldBe 0.125f +- 0.01f
    }
