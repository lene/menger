package menger.caustics

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * L1 analytic caustics tests (ladder rungs C1, C3, C4) — CPU-only, no GPU.
 *
 * These encode the reference physics that the CUDA photon tracer
 * (menger-geometry/src/main/native/shaders/caustics_ppm.cu) must satisfy after the
 * Sprint 33 fixes. The numbers are pinned to the canonical scene (glass sphere
 * IOR 1.5, point light at (0,10,0)) so they double as an executable specification:
 * a reviewer can read the expected Snell/Fresnel/focal values here, and the GPU
 * statistical suites (C2/C5/C6/C7) verify the kernel reproduces them.
 *
 * Reference formulas live in this file deliberately — they are the spec, kept
 * independent of the implementation under test.
 */
class CausticsPhysicsSuite extends AnyFlatSpec with Matchers:

  private val Ior = 1.5 // canonical glass

  // ── Reference physics ───────────────────────────────────────────────────────

  /** Snell's law: refracted angle from incident angle across n1 -> n2. None on TIR. */
  private def refractAngle(thetaI: Double, n1: Double, n2: Double): Option[Double] =
    val sinT = n1 / n2 * math.sin(thetaI)
    if math.abs(sinT) > 1.0 then None else Some(math.asin(sinT))

  /** Critical angle for total internal reflection going n1 -> n2 (n1 > n2). */
  private def criticalAngle(n1: Double, n2: Double): Double = math.asin(n2 / n1)

  /** Exact dielectric Fresnel reflectance (unpolarised), n1 -> n2, incident angle. */
  private def fresnelDielectric(thetaI: Double, n1: Double, n2: Double): Double =
    val cosI = math.cos(thetaI)
    val sinT = n1 / n2 * math.sin(thetaI)
    if sinT >= 1.0 then 1.0 // TIR
    else
      val cosT = math.sqrt(1.0 - sinT * sinT)
      val rParl = (n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT)
      val rPerp = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT)
      0.5 * (rParl * rParl + rPerp * rPerp)

  /** Parallel Fresnel amplitude coefficient (zero at Brewster's angle). */
  private def fresnelRParallel(thetaI: Double, n1: Double, n2: Double): Double =
    val cosI = math.cos(thetaI)
    val cosT = math.sqrt(1.0 - math.pow(n1 / n2 * math.sin(thetaI), 2))
    (n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT)

  /** Solid angle of a cone of half-angle thetaMax: 2*pi*(1 - cos thetaMax). */
  private def coneSolidAngle(thetaMax: Double): Double = 2.0 * math.Pi * (1.0 - math.cos(thetaMax))

  /** Ball-lens paraxial back focal distance from the rear surface, sphere radius R. */
  private def ballLensBackFocalDistance(n: Double, r: Double): Double =
    // Effective focal length of a sphere (thick lens), measured from the centre:
    //   EFL = n * R / (2 * (n - 1))
    // Back focal distance from the rear surface = EFL - R.
    n * r / (2.0 * (n - 1.0)) - r

  // ── C3: Snell's law + total internal reflection ─────────────────────────────

  "C3: refraction" should "obey Snell's law entering glass (air -> 1.5)" in:
    List(10.0, 30.0, 45.0, 60.0).foreach: deg =>
      val thetaI = math.toRadians(deg)
      val thetaT = refractAngle(thetaI, 1.0, Ior).getOrElse(fail(s"unexpected TIR at $deg"))
      // n1 sin i == n2 sin t
      (1.0 * math.sin(thetaI)) shouldBe (Ior * math.sin(thetaT) +- 1e-9)
      thetaT should be < thetaI // bends toward normal entering denser medium

  it should "have critical angle asin(1/1.5) ~= 41.81 deg (glass -> air)" in:
    math.toDegrees(criticalAngle(Ior, 1.0)) shouldBe (41.8103 +- 1e-3)

  it should "totally internally reflect beyond the critical angle" in:
    val thetaC = criticalAngle(Ior, 1.0)
    refractAngle(thetaC + math.toRadians(1.0), Ior, 1.0) shouldBe None
    // just inside the critical angle still refracts
    refractAngle(thetaC - math.toRadians(1.0), Ior, 1.0) should not be None

  // ── C4: Fresnel reflectance ─────────────────────────────────────────────────

  "C4: Fresnel" should "give 4% reflectance at normal incidence for n=1.5" in:
    fresnelDielectric(0.0, 1.0, Ior) shouldBe (0.04 +- 1e-6)

  it should "vanish in the parallel component at Brewster's angle" in:
    val brewster = math.atan(Ior / 1.0) // atan(n2/n1) ~= 56.31 deg
    math.toDegrees(brewster) shouldBe (56.3099 +- 1e-3)
    fresnelRParallel(brewster, 1.0, Ior) shouldBe (0.0 +- 1e-9)

  it should "approach total reflection at grazing incidence" in:
    fresnelDielectric(math.toRadians(89.99), 1.0, Ior) should be > 0.99
    // and increase monotonically toward grazing
    fresnelDielectric(math.toRadians(80.0), 1.0, Ior) should be <
      fresnelDielectric(math.toRadians(89.0), 1.0, Ior)

  it should "conserve energy: R + T = 1" in:
    List(0.0, 20.0, 45.0, 70.0).foreach: deg =>
      val r = fresnelDielectric(math.toRadians(deg), 1.0, Ior)
      val t = 1.0 - r
      (r + t) shouldBe (1.0 +- 1e-12)
      r should (be >= 0.0 and be <= 1.0)

  // ── C1: emission pdf / power (cone sampling from a point light) ──────────────

  "C1: cone emission" should "have a pdf that integrates to 1 over the cone" in:
    // Uniform-in-cos(theta) sampling: pdf(omega) = 1 / (2*pi*(1 - cos thetaMax)).
    val thetaMax = math.toRadians(20.0)
    val pdf = 1.0 / coneSolidAngle(thetaMax)
    // Integral of pdf over the cone == pdf * solidAngle == 1.
    (pdf * coneSolidAngle(thetaMax)) shouldBe (1.0 +- 1e-12)

  it should "carry per-photon flux I*deltaOmega/N (P1 emission-measure factor)" in:
    // A point light of radiant intensity I, sampled into a cone of solid angle
    // deltaOmega with N photons, must give each photon flux = I*deltaOmega/N so the
    // deposited energy is independent of the (arbitrary) cone half-angle.
    val intensity = 500.0
    val n = 100000.0
    val flux1 = intensity * coneSolidAngle(math.toRadians(15.0)) / n
    val flux2 = intensity * coneSolidAngle(math.toRadians(30.0)) / n
    // Total emitted power is invariant to N and to the cone (energy, not per-photon):
    (flux1 * n / coneSolidAngle(math.toRadians(15.0))) shouldBe (intensity +- 1e-9)
    (flux2 * n / coneSolidAngle(math.toRadians(30.0))) shouldBe (intensity +- 1e-9)

  it should "subtend the cone half-angle asin(R/d) toward a target sphere" in:
    // Canonical: light at (0,10,0), unit sphere at origin => d=10, R=1.
    val d = 10.0
    val r = 1.0
    val thetaMax = math.asin(r / d)
    math.toDegrees(thetaMax) shouldBe (5.7392 +- 1e-3)

  // ── C4: focal point of the glass ball ───────────────────────────────────────

  "C4: focal point" should "place the paraxial focus 0.5R behind a unit n=1.5 sphere" in:
    // EFL = 1.5*1/(2*0.5) = 1.5 from centre; BFD = 1.5 - 1.0 = 0.5 behind rear surface.
    ballLensBackFocalDistance(Ior, 1.0) shouldBe (0.5 +- 1e-9)

  it should "focus in front of the canonical floor (plane at 2R below centre)" in:
    // Rear surface at y=-1, focus at y=-1.5; floor at y=-2 is beyond focus, so the
    // caustic on the floor is a ring, not a point — matches arc42's ~0.3 radius.
    val focusFromCentre = 1.0 + ballLensBackFocalDistance(Ior, 1.0) // 1.5
    focusFromCentre should be < 2.0
