package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CameraPathSuite extends AnyFlatSpec with Matchers:

  private val eps = 1e-5f
  private def approx(a: Float, b: Float): Boolean = math.abs(a - b) < eps
  private def approxVec(a: Vec3, b: Vec3): Boolean =
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)

  // --- Bezier ---

  "Bezier.cubic" should "return p0 at t=0" in:
    val p0 = Vec3(1f, 2f, 3f)
    val p3 = Vec3(7f, 8f, 9f)
    approxVec(Bezier.cubic(p0, Vec3.Zero, Vec3.Zero, p3)(0f), p0) shouldBe true

  it should "return p3 at t=1" in:
    val p0 = Vec3(1f, 2f, 3f)
    val p3 = Vec3(7f, 8f, 9f)
    approxVec(Bezier.cubic(p0, Vec3.Zero, Vec3.Zero, p3)(1f), p3) shouldBe true

  it should "lie on the curve at t=0.5 for symmetric control points" in:
    // When P0=(-1,0,0), P1=(-1,1,0), P2=(1,1,0), P3=(1,0,0), midpoint is (0,0.75,0)
    val result = Bezier.cubic(
      Vec3(-1f, 0f, 0f), Vec3(-1f, 1f, 0f),
      Vec3( 1f, 1f, 0f), Vec3( 1f, 0f, 0f)
    )(0.5f)
    approx(result.x, 0f)    shouldBe true
    approx(result.y, 0.75f) shouldBe true
    approx(result.z, 0f)    shouldBe true

  // --- CameraPath endpoints ---

  "CameraPath" should "return the first position at t=0" in:
    val a = Vec3(0f, 3f, 6f)
    val b = Vec3(6f, 3f, 0f)
    val path = CameraPath(List(a, b))
    approxVec(path.positionAt(0f), a) shouldBe true

  it should "return the last position at t=1" in:
    val a = Vec3(0f, 3f, 6f)
    val b = Vec3(6f, 3f, 0f)
    val path = CameraPath(List(a, b))
    approxVec(path.positionAt(1f), b) shouldBe true

  it should "clamp t below 0 to the first position" in:
    val a = Vec3(1f, 0f, 0f)
    val b = Vec3(2f, 0f, 0f)
    approxVec(CameraPath(List(a, b)).positionAt(-0.5f), a) shouldBe true

  it should "clamp t above 1 to the last position" in:
    val a = Vec3(1f, 0f, 0f)
    val b = Vec3(2f, 0f, 0f)
    approxVec(CameraPath(List(a, b)).positionAt(1.5f), b) shouldBe true

  it should "interpolate between endpoints (midpoint inside bounding box)" in:
    val a = Vec3(0f, 0f, 0f)
    val b = Vec3(4f, 0f, 0f)
    val mid = CameraPath(List(a, b)).positionAt(0.5f)
    mid.x should be > 0f
    mid.x should be < 4f

  it should "pass through intermediate waypoints at their respective t values" in:
    val a = Vec3(0f, 0f, 0f)
    val m = Vec3(2f, 4f, 0f)
    val b = Vec3(4f, 0f, 0f)
    val path = CameraPath(List(a, m, b))
    approxVec(path.positionAt(0f),   a) shouldBe true
    approxVec(path.positionAt(0.5f), m) shouldBe true
    approxVec(path.positionAt(1f),   b) shouldBe true

  // --- Camera convenience ---

  it should "preserve the lookAt target in the returned Camera" in:
    val target = Vec3(1f, 2f, 3f)
    val path   = CameraPath(List(Vec3(0f, 5f, 10f), Vec3(10f, 5f, 0f)), lookAt = target)
    path.at(0.5f).lookAt shouldBe target

  it should "preserve the up vector in the returned Camera" in:
    val up   = Vec3(0f, 0f, 1f)
    val path = CameraPath(List(Vec3(0f, 0f, 5f), Vec3(5f, 0f, 0f)), up = up)
    path.at(0.3f).up shouldBe up

  // --- lookingAt constructor ---

  "CameraPath.lookingAt" should "build a path with a fixed look-at" in:
    val path = CameraPath.lookingAt(Vec3.Zero, Vec3(0f, 3f, 6f), Vec3(6f, 3f, 0f))
    path.lookAt shouldBe Vec3.Zero
    path.positions should have size 2

  // --- require ---

  it should "reject fewer than 2 positions" in:
    an[IllegalArgumentException] should be thrownBy CameraPath(List(Vec3(1f, 0f, 0f)))
