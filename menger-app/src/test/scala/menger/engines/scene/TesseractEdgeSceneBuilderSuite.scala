package menger.engines.scene

import menger.cli.ObjectSpecParser
import menger.spec.ObjectSpec
import menger.spec.Projection4DSpec
import menger.common.ProfilingConfig
import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Sprint 36 H1.5: `isClippedByEyeW` mirrors the eye_w clip every 4D CUDA closest-hit shader
  * applies (e.g. `hit_menger4d.cu`'s `rot.w >= m.eye_w - 1e-6f`), so the CPU edge-projection
  * path in TesseractEdgeSceneBuilder degrades the same way the GPU face path does instead of
  * handing Projection.apply a w-coordinate at or past eyeW.
  *
  * This is a defensive parity fix, not a fix for the "console error spam while rotating"
  * report: rotation preserves a vertex's 4D norm, so at the CLI defaults (eyeW=3.0, object
  * size 0.8-1.5) no achievable rotation reaches the clip — see the second test below, which
  * proves that bound empirically rather than asserting it. The real cause of the console
  * spam remains open (ManualTestNeedFixing.md section 2). */
class TesseractEdgeSceneBuilderSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private val builder = TesseractEdgeSceneBuilder(".")

  // Usability review 2026-09, session 2 (F44): an explicit default projection next to an
  // absent one projects identically, but (Some, None) was always rejected as incompatible.
  // `parse` fills in the default projection; only a DSL object without `projection` gives None.
  private def withoutProjection =
    ObjectSpecParser.parse("type=tesseract").toOption.get.copy(projection4D = None)

  "isCompatible" should "treat an explicit default projection like an absent one" in:
    val absent = withoutProjection
    val explicitDefault = absent.copy(projection4D = Some(Projection4DSpec.default))
    builder.isCompatible(explicitDefault, absent) shouldBe true
    builder.isCompatible(absent, explicitDefault) shouldBe true

  // menger#52: each 4D object is projected with its own projection.
  it should "accept a projection that differs from the default" in:
    val absent = withoutProjection
    val rotated = absent.copy(projection4D = Some(Projection4DSpec(rotXW = 20f)))
    builder.isCompatible(rotated, absent) shouldBe true

  "isClippedByEyeW" should "not clip a vertex safely inside eyeW" in:
    builder.isClippedByEyeW(Vector[4](0f, 0f, 0f, 0f), eyeW = 3.0f) shouldBe false

  it should "clip a vertex exactly at eyeW" in:
    builder.isClippedByEyeW(Vector[4](0f, 0f, 0f, 3.0f), eyeW = 3.0f) shouldBe true

  it should "clip a vertex past eyeW" in:
    builder.isClippedByEyeW(Vector[4](0f, 0f, 0f, 3.5f), eyeW = 3.0f) shouldBe true

  it should "not clip a vertex just short of eyeW" in:
    builder.isClippedByEyeW(Vector[4](0f, 0f, 0f, 2.999f), eyeW = 3.0f) shouldBe false

  "A tesseract's rotated vertices" should
    "never reach eyeW at CLI-default size and eyeW, for any rotation" in:
    // Rotation is norm-preserving, so the reachable |rotated w| for any vertex is bounded by
    // that vertex's 4D norm regardless of rotation angle. A tesseract's farthest vertex from
    // the origin is a corner at (±size/2, ±size/2, ±size/2, ±size/2), norm = size. That is
    // the empirical basis for downgrading this fix's claim above: it does not explain the
    // reported console spam at default parameters (eyeW=3.0, size 0.8-1.5).
    val size = 0.8f
    val corner = Vector[4](size / 2, size / 2, size / 2, size / 2)
    val maxPossibleW = math.sqrt(
      (0 until 4).map(i => corner(i).toDouble * corner(i)).sum
    )
    val eyeW = 3.0f
    maxPossibleW should be < eyeW.toDouble
    builder.isClippedByEyeW(corner, eyeW) shouldBe false

  it should "reach eyeW once size grows past it (the case this guard actually protects)" in:
    // A --size well past --eye-w is contrived but not rejected by CLI validation, and this is
    // where the eye_w clip earns its keep: without it, Projection.apply's denominator would
    // cross zero on some vertex under an ordinary rotation.
    val size = 8.0f
    val corner = Vector[4](size / 2, size / 2, size / 2, size / 2)
    val eyeW = 3.0f
    builder.isClippedByEyeW(corner, eyeW) shouldBe true

  // Usability review 2026-10, session 4 (F94): edges were built from the integer part of a
  // fractional level, so they stepped at each integer.
  private def sponge(level: Float) =
    ObjectSpecParser.parse(s"type=tesseract-sponge:level=$level:edge-radius=0.02").toOption.get

  "edgeLevels" should "draw one level at an integer level" in:
    TesseractEdgeSceneBuilder.edgeLevels(sponge(1f)).map((s, k) => (s.level, k)) shouldBe
      List((Some(1f), 1f))

  it should "cross-fade level n and n+1 at a fractional level that fits" in:
    val levels = TesseractEdgeSceneBuilder.edgeLevels(sponge(1.25f)).map((s, k) => (s.level, k))

    levels.map(_._1) shouldBe List(Some(1f), Some(2f))
    levels.map(_._2) shouldBe List(0.75f, 0.25f)

  it should "keep the integer part where level n+1 has too many edges (from 2.x)" in:
    TesseractEdgeSceneBuilder.edgeLevels(sponge(2.5f)).map((s, k) => (s.level, k)) shouldBe
      List((Some(2f), 1f))

  "fadedEdgeMaterial" should "scale alpha, and IOR above 1, by the share kept" in:
    val glass = menger.common.Material.Glass
    val faded = TesseractEdgeSceneBuilder.fadedEdgeMaterial(glass, 0.5f)

    faded.color.a shouldBe (glass.color.a * 0.5f +- 1e-6f)
    faded.ior shouldBe (1f + (glass.ior - 1f) * 0.5f +- 1e-6f)
    TesseractEdgeSceneBuilder.fadedEdgeMaterial(glass, 1f) shouldBe glass

  it should "leave an IOR of 1 alone" in:
    val matte = menger.common.Material.Matte
    TesseractEdgeSceneBuilder.fadedEdgeMaterial(matte, 0.5f).ior shouldBe matte.ior

  "minEdgeInstancesOf" should "count both levels of a cross-fade" in:
    // Below level 3 the lower bound is 0 (cheap to count exactly): both levels stay 0.
    TesseractEdgeSceneBuilder.minEdgeInstancesOf(sponge(1.5f)) shouldBe 0L
