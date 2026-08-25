package menger.engines.scene

import menger.dsl.Vec3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CurveCornerSharpeningSuite extends AnyFlatSpec with Matchers:

  "sharpenCorners" should "triple a 90-degree corner point (Sprint 36 #17)" in:
    val points = Vector(Vec3(0f, 0f, 0f), Vec3(1f, 0f, 0f), Vec3(1f, 1f, 0f))
    val widths = Vector(0.1f, 0.2f, 0.3f)
    val (sharpPoints, sharpWidths) = CurveCornerSharpening.sharpenCorners(points, widths)
    // Endpoints once, corner point 3x: 1 + 3 + 1 = 5.
    sharpPoints.length shouldBe 5
    sharpWidths.length shouldBe 5
    sharpPoints(1) shouldBe points(1)
    sharpPoints(2) shouldBe points(1)
    sharpPoints(3) shouldBe points(1)
    sharpWidths(1) shouldBe widths(1)
    sharpWidths(2) shouldBe widths(1)
    sharpWidths(3) shouldBe widths(1)

  it should "leave a gentle bend below the sharpness threshold untouched" in:
    // ~11.4-degree turn (1 unit forward, 0.2 unit sideways) -- well under the 60-degree
    // threshold that separates hilbert3d's right angles from organic branch bends.
    val points = Vector(Vec3(0f, 0f, 0f), Vec3(1f, 0f, 0f), Vec3(2f, 0.2f, 0f))
    val widths = Vector(0.1f, 0.1f, 0.1f)
    val (sharpPoints, sharpWidths) = CurveCornerSharpening.sharpenCorners(points, widths)
    sharpPoints shouldBe points
    sharpWidths shouldBe widths

  it should "leave a perfectly straight run untouched" in:
    val points = Vector(Vec3(0f, 0f, 0f), Vec3(1f, 0f, 0f), Vec3(2f, 0f, 0f), Vec3(3f, 0f, 0f))
    val widths = Vector(0.1f, 0.1f, 0.1f, 0.1f)
    val (sharpPoints, sharpWidths) = CurveCornerSharpening.sharpenCorners(points, widths)
    sharpPoints shouldBe points
    sharpWidths shouldBe widths

  it should "pass through runs shorter than 3 points unchanged" in:
    val points = Vector(Vec3(0f, 0f, 0f), Vec3(1f, 0f, 0f))
    val widths = Vector(0.1f, 0.2f)
    CurveCornerSharpening.sharpenCorners(points, widths) shouldBe (points, widths)

  it should "sharpen multiple consecutive corners independently" in:
    // A zig-zag of three 90-degree corners.
    val points = Vector(
      Vec3(0f, 0f, 0f), Vec3(1f, 0f, 0f), Vec3(1f, 1f, 0f), Vec3(2f, 1f, 0f), Vec3(2f, 2f, 0f)
    )
    val widths = Vector.fill(points.length)(0.1f)
    val (sharpPoints, _) = CurveCornerSharpening.sharpenCorners(points, widths)
    // 2 endpoints once + 3 interior corners x3 = 2 + 9 = 11.
    sharpPoints.length shouldBe 11
