package menger.objects.higher_d

import menger.engines.WithAnimation
import menger.spec.ObjectSpec
import menger.spec.Projection4DSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** menger#164 (from menger#65, usability review 2026-10, session 3, F81): a per-axis scale applied
  * before rotation and projection, so a scene can grow a tesseract out of a flat cube and an
  * n-cube out of an (n-1)-cube. */
class AxisScaledMeshSpec extends AnyFlatSpec with Matchers:

  private val Size = 2f
  private val Half = 0.5f

  private def scalesWith(dimension: Int, axis: Int, factor: Float): IndexedSeq[Float] =
    IndexedSeq.tabulate(dimension)(a => if a == axis then factor else 1f)

  private def coordinates[N <: Int & Singleton](mesh: Mesh[N]): Seq[IndexedSeq[Float]] =
    mesh.vertices.map(_.toIndexedSeq)

  private def faceCoordinates[N <: Int & Singleton](mesh: Mesh[N]): Seq[IndexedSeq[Float]] =
    mesh.faces.flatMap(_.asSeq).map(_.toIndexedSeq)

  "AxisScaledMesh" should "leave a mesh at the identity scale as it is, for N = 4..8" in:
    def identityHolds[N <: Int & Singleton](using n: ValueOf[N]): Unit =
      val cube = NCube[N](Size)
      AxisScaledMesh.of(cube, IndexedSeq.fill(n.value)(1f)) shouldBe theSameInstanceAs(cube)
      coordinates(AxisScaledMesh(cube, IndexedSeq.fill(n.value)(1f))) shouldBe coordinates(cube)
    identityHolds[4]; identityHolds[5]; identityHolds[6]; identityHolds[7]; identityHolds[8]

  it should "change only coordinate k when axis k is scaled, for N = 4..8" in:
    def onlyAxisChanges[N <: Int & Singleton](using n: ValueOf[N]): Unit =
      val cube = NCube[N](Size)
      (0 until n.value).foreach { axis =>
        val scaled = AxisScaledMesh(cube, scalesWith(n.value, axis, Half))
        val expected = coordinates(cube).map(v => v.updated(axis, v(axis) * Half))
        coordinates(scaled) shouldBe expected
        faceCoordinates(scaled) shouldBe
          faceCoordinates(cube).map(v => v.updated(axis, v(axis) * Half))
      }
    onlyAxisChanges[4]; onlyAxisChanges[5]; onlyAxisChanges[6]
    onlyAxisChanges[7]; onlyAxisChanges[8]

  it should "collapse the axis at scale 0, for N = 4..8" in:
    def collapses[N <: Int & Singleton](using n: ValueOf[N]): Unit =
      val cube = NCube[N](Size)
      val last = n.value - 1
      val flat = AxisScaledMesh(cube, scalesWith(n.value, last, 0f))
      coordinates(flat).map(_(last)).toSet shouldBe Set(0f)
      faceCoordinates(flat).map(_(last)).toSet shouldBe Set(0f)
      (flat.faces.size, flat.vertsPerFace) shouldBe (cube.faces.size, cube.vertsPerFace)
    collapses[4]; collapses[5]; collapses[6]; collapses[7]; collapses[8]

  it should "scale a tesseract's w as the old w-scale did" in:
    val tesseract = Tesseract(size = Size)
    val half = AxisScaledMesh(tesseract, IndexedSeq(1f, 1f, 1f, Half))
    half.vertices.map(v => (v(0), v(1), v(2), v(3))) shouldBe
      tesseract.vertices.map(v => (v(0), v(1), v(2), v(3) * Half))

  it should "reject a negative factor and a wrong length" in:
    an[IllegalArgumentException] should be thrownBy
      AxisScaledMesh(Tesseract(), IndexedSeq(1f, -1f, 1f, 1f))
    an[IllegalArgumentException] should be thrownBy
      AxisScaledMesh(Tesseract(), IndexedSeq(1f, 1f, 1f))

  it should "check the length of the scales before the identity shortcut" in:
    an[IllegalArgumentException] should be thrownBy
      AxisScaledMesh.of(Tesseract(), IndexedSeq(1f, 1f, 1f))
    AxisScaledMesh.nonIdentity(IndexedSeq(1f, 1f, 1f, 1f)) shouldBe None
    AxisScaledMesh.nonIdentity(IndexedSeq(1f, 1f, 1f, 0f)) should not be empty

  "ObjectSpec" should "reject a scale factor that is negative or not finite" in:
    Seq(-1f, Float.NaN, Float.PositiveInfinity).foreach { bad =>
      an[IllegalArgumentException] should be thrownBy
        ObjectSpec("tesseract", scale = Some(IndexedSeq(1f, 1f, 1f, bad)))
    }

  it should "reject a scale of the wrong length when the scales are read" in:
    val refusal = the[IllegalArgumentException] thrownBy
      ObjectSpec("tesseract", scale = Some(IndexedSeq(1f, 1f, 1f))).axisScales(4)
    refusal.getMessage should (include("3 factors") and include("4"))

  "ObjectSpec.axisScales" should "default to all 1 and read the scale" in:
    ObjectSpec("tesseract").axisScales(4) shouldBe IndexedSeq(1f, 1f, 1f, 1f)
    ObjectSpec("tesseract", scale = Some(IndexedSeq(1f, 1f, 1f, Half))).axisScales(4) shouldBe
      IndexedSeq(1f, 1f, 1f, Half)

  "WithAnimation.specsDifferAtMostIn4DProjection" should "not count a scale change as a view" in:
    val spec = ObjectSpec("tesseract")
    val scaled = spec.copy(scale = Some(IndexedSeq(1f, 1f, 1f, Half)))
    WithAnimation.specsDifferAtMostIn4DProjection(List(spec), List(scaled)) shouldBe false
    val turned = spec.copy(projection = Some(Projection4DSpec(rotXW = 40f)))
    WithAnimation.specsDifferAtMostIn4DProjection(List(spec), List(turned)) shouldBe true
