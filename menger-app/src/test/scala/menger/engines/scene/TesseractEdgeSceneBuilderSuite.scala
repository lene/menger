package menger.engines.scene

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
