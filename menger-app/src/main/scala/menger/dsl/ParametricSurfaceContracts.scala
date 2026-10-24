package menger.dsl

import menger.objects.higher_d.InvariantFinding

/** Runtime contracts for the two free-form/lambda DSL object types, which `require()` alone
  * cannot check: a hostile or careless `f: (Float, Float) => Vec3` can return NaN/Inf, a
  * surface whose `closedU`/`closedV` flags don't match what the function actually does at the
  * seam, or a degenerate (zero-area / zero-length) shape -- all of which rendered silently
  * wrong or crashed downstream instead of being caught here (usability review 2026-09, T3#13).
  *
  * Samples on a small fixed grid rather than the object's full `uSteps`/`vSteps` (up to
  * [[ResourceLimits.parametricSurfaceMaxSamples]]) -- this runs as part of `SceneValidator`'s
  * lint pass, not the actual tessellation, so it only needs to catch a bad `f`, not reproduce
  * the mesh.
  */
object ParametricSurfaceContracts:

  private val GridSamples = 8
  private val FiniteInvariant = "parametric-surface-finite"
  private val SeamInvariant = "parametric-surface-seam"
  private val AreaInvariant = "parametric-surface-area"
  private val ArcLengthInvariant = "curve-arc-length"

  def check(surface: ParametricSurface): List[InvariantFinding] =
    val (uMin, uMax) = surface.uRange
    val (vMin, vMax) = surface.vRange
    val us = (0 to GridSamples).map(i => uMin + (uMax - uMin) * i / GridSamples)
    val vs = (0 to GridSamples).map(j => vMin + (vMax - vMin) * j / GridSamples)
    val samples = for v <- vs; u <- us yield (u, v, surface.f(u, v))

    samples.collectFirst { case (u, v, p) if !isFinite(p) =>
      InvariantFinding(FiniteInvariant, s"f($u, $v) = $p is not finite (NaN or Inf)")
    } match
      case Some(finding) => List(finding)
      case None =>
        val extent = boundingRadius(samples.map(_._3))
        val tolerance = math.max(extent, 1e-6f) * 1e-3f

        val seamU =
          if surface.closedU then
            vs.find(v => distance(surface.f(uMin, v), surface.f(uMax, v)) > tolerance).map(v =>
              InvariantFinding(SeamInvariant,
                s"closedU is declared but f($uMin, $v) and f($uMax, $v) do not coincide"))
          else None

        val seamV =
          if surface.closedV then
            us.find(u => distance(surface.f(u, vMin), surface.f(u, vMax)) > tolerance).map(u =>
              InvariantFinding(SeamInvariant,
                s"closedV is declared but f($u, $vMin) and f($u, $vMax) do not coincide"))
          else None

        val areaFinding =
          if approximateArea(us, vs, surface.f) <= 0f then
            Some(InvariantFinding(AreaInvariant,
              "surface has zero or negative area -- f may be degenerate (e.g. constant)"))
          else None

        List(seamU, seamV, areaFinding).flatten

  def check(curve: Curve): List[InvariantFinding] =
    val pts = if curve.closed then curve.points :+ curve.points.head else curve.points
    val arcLength = pts.sliding(2).collect { case Seq(a, b) => distance(a, b) }.sum
    if arcLength <= 0f then
      List(InvariantFinding(ArcLengthInvariant,
        "curve has zero arc length -- all control points coincide"))
    else Nil

  private def isFinite(p: Vec3): Boolean =
    p.x.isFinite && p.y.isFinite && p.z.isFinite

  private def distance(a: Vec3, b: Vec3): Float =
    val dx = a.x - b.x; val dy = a.y - b.y; val dz = a.z - b.z
    math.sqrt((dx * dx + dy * dy + dz * dz).toDouble).toFloat

  private def boundingRadius(points: Seq[Vec3]): Float =
    if points.isEmpty then 0f
    else points.map(p => math.sqrt((p.x * p.x + p.y * p.y + p.z * p.z).toDouble).toFloat).max

  /** Sum of the per-grid-cell parallelogram area (`|f_u x f_v| * du * dv`), the same
    * cross-product-of-finite-differences formula [[menger.objects.ParametricTessellator]]
    * uses for normals, evaluated on this checker's own coarse grid instead of the
    * tessellator's full one. */
  private def approximateArea(
    us: Seq[Float], vs: Seq[Float], f: (Float, Float) => Vec3
  ): Float =
    val du = if us.size > 1 then us(1) - us(0) else 1f
    val dv = if vs.size > 1 then vs(1) - vs(0) else 1f
    val epsilon = 1e-4f * math.max(math.abs(du), math.abs(dv)).max(1e-6f)
    val cellAreas = for
      v <- vs
      u <- us
    yield
      val pu1 = f(u + epsilon, v); val pu0 = f(u - epsilon, v)
      val pv1 = f(u, v + epsilon); val pv0 = f(u, v - epsilon)
      val dxu = (pu1.x - pu0.x) / (2 * epsilon); val dyu = (pu1.y - pu0.y) / (2 * epsilon); val dzu = (pu1.z - pu0.z) / (2 * epsilon)
      val dxv = (pv1.x - pv0.x) / (2 * epsilon); val dyv = (pv1.y - pv0.y) / (2 * epsilon); val dzv = (pv1.z - pv0.z) / (2 * epsilon)
      val crossX = dyu * dzv - dzu * dyv
      val crossY = dzu * dxv - dxu * dzv
      val crossZ = dxu * dyv - dyu * dxv
      val cellArea = math.sqrt((crossX * crossX + crossY * crossY + crossZ * crossZ).toDouble) * du * dv
      if cellArea.isFinite then cellArea else 0.0
    cellAreas.sum.toFloat
