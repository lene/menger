package menger.dsl

import scala.language.implicitConversions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CurveSuite extends AnyFlatSpec with Matchers:

  private val fourPoints = Seq(Vec3(0f, 0f, 0f), Vec3(1f, 0f, 0f), Vec3(1f, 1f, 0f), Vec3(0f, 1f, 0f))

  "Curve" should "be constructible with minimal params" in:
    val curve = Curve(points = fourPoints)
    curve.points shouldBe fourPoints
    curve.radius shouldBe 0.05f
    curve.closed shouldBe false
    curve.material shouldBe None

  it should "reject fewer than 4 control points" in:
    an[IllegalArgumentException] should be thrownBy Curve(points = Seq(Vec3.Zero, Vec3(1f, 0f, 0f)))

  it should "reject non-positive radius" in:
    an[IllegalArgumentException] should be thrownBy Curve(points = fourPoints, radius = 0f)
    an[IllegalArgumentException] should be thrownBy Curve(points = fourPoints, radius = -0.1f)

  it should "reject NaN in control point coordinates" in:
    val nanPoint = fourPoints.updated(1, Vec3(Float.NaN, 0f, 0f))
    an[IllegalArgumentException] should be thrownBy Curve(points = nanPoint)

  it should "reject Inf in control point coordinates" in:
    val infPoint = fourPoints.updated(2, Vec3(0f, Float.PositiveInfinity, 0f))
    an[IllegalArgumentException] should be thrownBy Curve(points = infPoint)

  it should "reject NaN radius" in:
    an[IllegalArgumentException] should be thrownBy Curve(points = fourPoints, radius = Float.NaN)

  "Curve.toObjectSpec" should "produce objectType curve" in:
    val spec = Curve(points = fourPoints).toObjectSpec
    spec.objectType shouldBe "curve"

  it should "embed flattened xyz control points in curveData" in:
    val spec = Curve(points = fourPoints).toObjectSpec
    val cd = spec.curveData
    cd shouldBe defined
    cd.get.points.length shouldBe fourPoints.size * 3
    cd.get.points.length % 3 shouldBe 0

  it should "set widths length equal to control point count" in:
    val spec = Curve(points = fourPoints, radius = 0.1f).toObjectSpec
    val cd = spec.curveData.get
    cd.widths.length shouldBe fourPoints.size
    cd.widths.forall(_ == 0.1f) shouldBe true

  it should "use per-point radii when provided" in:
    val radii = Seq(0.1f, 0.2f, 0.3f, 0.4f)
    val spec = Curve(points = fourPoints, radii = Some(radii)).toObjectSpec
    spec.curveData.get.widths.toSeq shouldBe radii

  it should "append 3 wrap points when closed is true" in:
    val spec = Curve(points = fourPoints, closed = true).toObjectSpec
    val cd = spec.curveData.get
    val expectedPoints = fourPoints.size + 3
    cd.points.length shouldBe expectedPoints * 3
    cd.widths.length shouldBe expectedPoints

  it should "carry xyz values faithfully" in:
    val pts = Seq(Vec3(1f, 2f, 3f), Vec3(4f, 5f, 6f), Vec3(7f, 8f, 9f), Vec3(10f, 11f, 12f))
    val cd = Curve(points = pts).toObjectSpec.curveData.get
    cd.points.toSeq shouldBe Seq(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)

  "Curve.circle" should "produce at least 4 control points" in:
    val c = Curve.circle(radius = 1f, tubeRadius = 0.05f)
    c.points.size should be >= 4

  it should "produce a closed curve" in:
    Curve.circle(radius = 1f, tubeRadius = 0.05f).closed shouldBe true

  "Curve.helix" should "produce at least 4 control points" in:
    val h = Curve.helix(turns = 1, radius = 1f, pitch = 0.5f, tubeRadius = 0.05f)
    h.points.size should be >= 4

  it should "produce an open curve" in:
    Curve.helix(turns = 1, radius = 1f, pitch = 0.5f, tubeRadius = 0.05f).closed shouldBe false
