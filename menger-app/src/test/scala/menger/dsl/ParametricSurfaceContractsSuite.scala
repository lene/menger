package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ParametricSurfaceContractsSuite extends AnyFlatSpec with Matchers:

  private def flatPlane(u: Float, v: Float): Vec3 = Vec3(u, 0f, v)

  private def sphere(u: Float, v: Float): Vec3 =
    Vec3(math.sin(v).toFloat * math.cos(u).toFloat, math.cos(v).toFloat, math.sin(v).toFloat * math.sin(u).toFloat)

  "ParametricSurfaceContracts.check(ParametricSurface)" should
    "find nothing wrong with a genuine open surface" in:
      val surface = ParametricSurface(f = flatPlane, uRange = (0f, 1f), vRange = (0f, 1f))
      ParametricSurfaceContracts.check(surface) shouldBe empty

  it should "find nothing wrong with a genuine closed-in-u surface (a sphere's longitude)" in:
    val surface = ParametricSurface(
      f = sphere,
      uRange = (0f, 2f * math.Pi.toFloat), vRange = (0.01f, math.Pi.toFloat - 0.01f),
      closedU = true
    )
    ParametricSurfaceContracts.check(surface) shouldBe empty

  it should "flag a non-finite sample" in:
    val surface = ParametricSurface(f = (_, v) => Vec3(Float.NaN, 0f, v))
    val findings = ParametricSurfaceContracts.check(surface)
    findings.map(_.invariant) should contain("parametric-surface-finite")

  it should "flag closedU declared when the u-seam does not actually coincide" in:
    val surface = ParametricSurface(f = (u, v) => Vec3(u, 0f, v), closedU = true)
    val findings = ParametricSurfaceContracts.check(surface)
    findings.map(_.invariant) should contain("parametric-surface-seam")

  it should "flag closedV declared when the v-seam does not actually coincide" in:
    val surface = ParametricSurface(f = (u, v) => Vec3(u, 0f, v), closedV = true)
    val findings = ParametricSurfaceContracts.check(surface)
    findings.map(_.invariant) should contain("parametric-surface-seam")

  it should "flag a degenerate (constant) surface as zero-area" in:
    val surface = ParametricSurface(f = (_, _) => Vec3(1f, 1f, 1f))
    val findings = ParametricSurfaceContracts.check(surface)
    findings.map(_.invariant) should contain("parametric-surface-area")

  "ParametricSurfaceContracts.check(Curve)" should "find nothing wrong with a genuine curve" in:
    val curve = Curve(points = Seq(Vec3(0f, 0f, 0f), Vec3(1f, 0f, 0f), Vec3(2f, 0f, 0f), Vec3(3f, 0f, 0f)))
    ParametricSurfaceContracts.check(curve) shouldBe empty

  it should "flag a curve whose control points all coincide" in:
    val p = Vec3(1f, 2f, 3f)
    val curve = Curve(points = Seq(p, p, p, p))
    val findings = ParametricSurfaceContracts.check(curve)
    findings.map(_.invariant) should contain("curve-arc-length")
