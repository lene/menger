package menger

import io.github.lene.optix.UniformRenderException
import menger.engines.ScreenshotFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ScreenshotFactorySuite extends AnyFlatSpec with Matchers:

  // sanitizePath is the path sanitiser used by saveScreenshot (full paths,
  // forward slashes kept, `..` rejected).

  "sanitizePath" should "allow forward slashes for subdirectories" in:
    ScreenshotFactory.sanitizePath("output/test.png") should be("output/test.png")

  it should "allow nested subdirectories" in:
    ScreenshotFactory.sanitizePath("a/b/c/test.png") should be("a/b/c/test.png")

  it should "reject path traversal with .." in:
    an[IllegalArgumentException] should be thrownBy ScreenshotFactory.sanitizePath("../etc/passwd")

  it should "reject hidden path traversal" in:
    an[IllegalArgumentException] should be thrownBy ScreenshotFactory.sanitizePath("foo/../bar/test.png")

  it should "preserve absolute paths with leading slash" in:
    ScreenshotFactory.sanitizePath("/absolute/path.png") should be("/absolute/path.png")

  it should "add png extension if missing" in:
    ScreenshotFactory.sanitizePath("output/test") should be("output/test.png")

  it should "preserve existing png extension" in:
    ScreenshotFactory.sanitizePath("output/test.png") should be("output/test.png")

  it should "preserve pfm extension for linear float output" in:
    ScreenshotFactory.sanitizePath("output/test.pfm") should be("output/test.pfm")

  it should "remove dangerous characters but keep path separators" in:
    ScreenshotFactory.sanitizePath("output/test<script>.png") should be("output/testscript.png")

  "checkRgbaForSave" should "still reject a uniform denoised render" in:
    val pixels = Array.fill[Byte](4 * 4 * 4)(127.toByte)
    val ex = intercept[UniformRenderException]:
      ScreenshotFactory.checkRgbaForSave(
        pixels,
        width = 4,
        height = 4,
        path = "denoised.png",
        allow = false
      )
    ex.getMessage should include("Pass --allow-uniform-render")

  it should "allow a uniform denoised render when explicitly configured" in:
    val pixels = Array.fill[Byte](4 * 4 * 4)(127.toByte)
    noException should be thrownBy ScreenshotFactory.checkRgbaForSave(
      pixels,
      width = 4,
      height = 4,
      path = "denoised.png",
      allow = true
    )
