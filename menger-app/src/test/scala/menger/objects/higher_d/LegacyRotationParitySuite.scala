// Declared in menger.geometry, not next to its subject, to reach private[geometry]
// MengerRenderer.composeRotationXwYwZw.
package menger.geometry

import scala.util.Random

import menger.common.Vector
import menger.objects.Matrix
import menger.objects.higher_d.Mesh4D
import menger.objects.higher_d.Pentachoron
import menger.objects.higher_d.Plane
import menger.objects.higher_d.Projection
import menger.objects.higher_d.Rotation
import menger.objects.higher_d.Tesseract
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Pins the 4D rotation matrices, the 4D perspective projection and the 4D mesh edges bit for
  * bit, with verbatim copies of the code as it was before the nD generalisation (menger#162). */
class LegacyRotationParitySuite extends AnyFlatSpec with Matchers:

  private val GpuParityTolerance = 1e-6f
  private val randomAngleCount = 12
  private val randomSeed = 162L

  private val allPlanes =
    Seq(Plane.xy, Plane.xz, Plane.xw, Plane.yz, Plane.yw, Plane.zw)

  private val angleGrid: Seq[Float] =
    val fixed = Seq(
      -1080f, -720f, -361f, -180f, -90f, -45f, -1f, 0f, 0.5f, 1f, 10f, 30f, 45f, 60f, 90f, 120f,
      135f, 180f, 270f, 359f, 360f, 361f, 405f, 720f, 1000f
    )
    val random = Random(randomSeed)
    fixed ++ Seq.fill(randomAngleCount)((random.nextFloat() - 0.5f) * 2000f)

  private def bits(f: Float): Int = java.lang.Float.floatToRawIntBits(f)
  private def bits(values: Iterable[Float]): Seq[Int] = values.map(bits).toSeq

  // verbatim copy of Rotation.matrix(row, col, angle) as of baseline 85197922
  private def legacyMatrix(row: Int, col: Int, angle: Float): Matrix[4] =
    val cosTheta: Float = math.cos(angle.toRadians).toFloat
    val sinTheta: Float = math.sin(angle.toRadians).toFloat
    val m = Array.fill(4, 4)(0f)
    for i <- 0 to 3 do m(i)(i) = 1f
    m(row)(row) = cosTheta
    m(col)(row) = -sinTheta
    m(row)(col) = sinTheta
    m(col)(col) = cosTheta
    Matrix[4](m.flatten)

  // verbatim copy of the XW * YW * ZW composition in Rotation.apply(degreesXW, YW, ZW, pivot)
  private def legacyXwYwZw(xw: Float, yw: Float, zw: Float): Matrix[4] =
    legacyMatrix(0, 3, xw) * legacyMatrix(1, 3, yw) * legacyMatrix(2, 3, zw)

  private def shouldBeBitIdentical(actual: Matrix[4], expected: Matrix[4], clue: String): Unit =
    withClue(clue) { bits(actual.m) shouldBe bits(expected.m) }

  "Rotation(xw, yw, zw)" should "produce the legacy matrix bit for bit" in:
    for xw <- angleGrid; yw <- angleGrid.take(8); zw <- angleGrid.drop(8).take(8) do
      shouldBeBitIdentical(
        Rotation(xw, yw, zw).transformationMatrix, legacyXwYwZw(xw, yw, zw), s"($xw, $yw, $zw): "
      )

  it should "produce the legacy matrix bit for bit when only one angle is set" in:
    for a <- angleGrid do
      shouldBeBitIdentical(
        Rotation(a, 0f, 0f).transformationMatrix, legacyXwYwZw(a, 0f, 0f), s"xw $a: "
      )
      shouldBeBitIdentical(
        Rotation(0f, a, 0f).transformationMatrix, legacyXwYwZw(0f, a, 0f), s"yw $a: "
      )
      shouldBeBitIdentical(
        Rotation(0f, 0f, a).transformationMatrix, legacyXwYwZw(0f, 0f, a), s"zw $a: "
      )

  "Rotation(plane, angle)" should "produce the legacy plane matrix for all six planes" in:
    for plane <- allPlanes; a <- angleGrid do
      shouldBeBitIdentical(
        Rotation(plane, a).transformationMatrix, legacyMatrix(plane.i, plane.j, a),
        s"$plane $a: "
      )

  it should "keep the pivot point in the pivoted variant" in:
    val pivot = Vector[4](1f, -2f, 3f, 0.5f)
    for plane <- allPlanes; a <- angleGrid.take(10) do
      val rotation = Rotation(plane, a, pivot)
      shouldBeBitIdentical(
        rotation.transformationMatrix, legacyMatrix(plane.i, plane.j, a), s"$plane $a: "
      )
      rotation.pivotPoint shouldBe pivot

  "Rotation applied to a point" should "equal the legacy matrix applied around the pivot" in:
    val random = Random(randomSeed)
    val pivot = Vector[4](0.5f, -1f, 2f, 0.25f)
    for _ <- 0 until 40 do
      val p = Vector[4](Seq.fill(4)((random.nextFloat() - 0.5f) * 20f)*)
      val (xw, yw, zw) = (random.nextFloat() * 720f, random.nextFloat() * 720f, 90f)
      val expected = legacyXwYwZw(xw, yw, zw)(p - pivot) + pivot
      bits(Rotation(xw, yw, zw, pivot)(p).toIndexedSeq) shouldBe bits(expected.toIndexedSeq)

  "Projection.apply" should "equal its formula bit for bit" in:
    val random = Random(randomSeed)
    for _ <- 0 until 100 do
      val screenW = 0.5f + random.nextFloat() * 2f
      val eyeW = screenW + 0.5f + random.nextFloat() * 5f
      val p = Vector[4](Seq.fill(4)((random.nextFloat() - 0.5f) * 2f)*)
      val factor = (eyeW - screenW) / (eyeW - p(3))
      val expected = Seq(p(0) * factor, p(1) * factor, p(2) * factor)
      bits(Projection(eyeW, screenW)(p).toIndexedSeq) shouldBe bits(expected)

  // verbatim copy of Mesh4D.edges as of baseline 85197922
  private def legacyEdges(mesh: Mesh4D): Set[(Vector[4], Vector[4])] =
    type VKey = (Float, Float, Float, Float)
    def vkey(v: Vector[4]): VKey = (v(0), v(1), v(2), v(3))
    def canonical(a: Vector[4], b: Vector[4]): (Vector[4], Vector[4]) =
      if summon[Ordering[VKey]].lteq(vkey(a), vkey(b)) then (a, b) else (b, a)
    mesh.faces.flatMap { f =>
      (0 until f.vertsPerFace).map { i =>
        canonical(f(i), f((i + 1) % f.vertsPerFace))
      }
    }.toSet

  private def edgeBits(edges: Set[(Vector[4], Vector[4])]): Set[(Seq[Int], Seq[Int])] =
    edges.map((a, b) => (bits(a.toIndexedSeq), bits(b.toIndexedSeq)))

  "Mesh4D.edges" should "equal the legacy edges for a Tesseract, in pair order" in:
    val tesseract = Tesseract(1.5f)
    val edges = tesseract.edges
    edges.size shouldBe 32
    edgeBits(edges) shouldBe edgeBits(legacyEdges(tesseract))

  it should "equal the legacy edges for a Pentachoron, in pair order" in:
    val pentachoron = Pentachoron(2f)
    val edges = pentachoron.edges
    edges.size shouldBe 10
    edgeBits(edges) shouldBe edgeBits(legacyEdges(pentachoron))

  "Rotation.apply" should "agree with the GPU composition R_xw * (R_yw * R_zw) within tolerance" in:
    for xw <- angleGrid; yw <- angleGrid.take(8); zw <- angleGrid.drop(8).take(8) do
      val gpu = MengerRenderer.composeRotationXwYwZw(xw, yw, zw)
      val scala = Rotation(xw, yw, zw).transformationMatrix.m
      withClue(s"($xw, $yw, $zw): ") {
        scala.zip(gpu).foreach((s, g) => s shouldBe g +- GpuParityTolerance)
      }
