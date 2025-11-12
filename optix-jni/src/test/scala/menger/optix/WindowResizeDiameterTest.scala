package menger.optix

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import com.typesafe.scalalogging.LazyLogging

class WindowResizeDiameterTest extends AnyFlatSpec with Matchers with LazyLogging {

  /**
   * Measure the vertical diameter of the sphere by scanning the center column.
   * Returns the number of pixels where the sphere is visible.
   */
  private def measureVerticalDiameter(img: Array[Byte], width: Int, height: Int): Int = {
    val centerX = width / 2
    var count = 0
    for (y <- 0 until height) {
      val idx = (y * width + centerX) * 4
      val r = img(idx) & 0xFF
      val g = img(idx + 1) & 0xFF
      val b = img(idx + 2) & 0xFF
      // Detect red sphere pixels: r>50 (not too dark), and red is at least 2x green and blue
      if (r > 50 && r > g * 2 && r > b * 2) {
        count += 1
      }
    }
    count
  }

  /**
   * Measure the horizontal diameter of the sphere by scanning the center row.
   * Returns the number of pixels where the sphere is visible.
   */
  private def measureHorizontalDiameter(img: Array[Byte], width: Int, height: Int): Int = {
    val centerY = height / 2
    var count = 0
    for (x <- 0 until width) {
      val idx = (centerY * width + x) * 4
      val r = img(idx) & 0xFF
      val g = img(idx + 1) & 0xFF
      val b = img(idx + 2) & 0xFF
      // Detect red sphere pixels: r>50 (not too dark), and red is at least 2x green and blue
      if (r > 50 && r > g * 2 && r > b * 2) {
        count += 1
      }
    }
    count
  }

  /**
   * Fixture: Create a renderer with a red sphere at origin and render at 800x800.
   * Returns (renderer, baseline vertical diameter, baseline horizontal diameter).
   */
  private def createBaselineRender(): (OptiXRenderer, Int, Int) = {
    val renderer = new OptiXRenderer()
    renderer.initialize() shouldBe true

    // Set up a red sphere at origin
    renderer.setSphere(0f, 0f, 0f, 1.0f)
    renderer.setSphereColor(1f, 0f, 0f, 1f)

    val eye = Array(0f, 0f, 5f)
    val lookAt = Array(0f, 0f, 0f)
    val up = Array(0f, 1f, 0f)
    val fov = 45f

    // Render baseline 800x800 image
    renderer.updateImageDimensions(800, 800)
    renderer.setCamera(eye, lookAt, up, fov)
    val img = renderer.render(800, 800)
    TestUtilities.savePPM("window_resize_diameter_baseline_800x800.ppm", img, 800, 800)

    val vDiam = measureVerticalDiameter(img, 800, 800)
    val hDiam = measureHorizontalDiameter(img, 800, 800)
    logger.info(s"Baseline 800x800: vertical=${vDiam}px, horizontal=${hDiam}px")

    // Sphere should be circular in square aspect
    Math.abs(vDiam - hDiam) should be < 5

    (renderer, vDiam, hDiam)
  }

  "OptiXRenderer" should "scale sphere when window width changes" in {
    assume(OptiXRenderer.isLibraryLoaded, "OptiX library not available")

    val (renderer, baselineVDiam, baselineHDiam) = createBaselineRender()

    try {
      val eye = Array(0f, 0f, 5f)
      val lookAt = Array(0f, 0f, 0f)
      val up = Array(0f, 1f, 0f)
      val fov = 45f

      // Render at 1600x800 (wide 2:1 - width doubled)
      renderer.updateImageDimensions(1600, 800)
      renderer.setCamera(eye, lookAt, up, fov)
      val img = renderer.render(1600, 800)
      TestUtilities.savePPM("window_resize_horizontal_1600x800.ppm", img, 1600, 800)

      val vDiam = measureVerticalDiameter(img, 1600, 800)
      val hDiam = measureHorizontalDiameter(img, 1600, 800)
      logger.info(s"Horizontal resize 1600x800: vertical=${vDiam}px, horizontal=${hDiam}px")

      // With fixed vertical FOV, sphere size stays constant regardless of aspect ratio
      Math.abs(vDiam - baselineVDiam) should be < 10
      Math.abs(hDiam - baselineHDiam) should be < 10

      // Sphere should still be circular
      Math.abs(vDiam - hDiam) should be < 5

    } finally {
      renderer.dispose()
    }
  }

  it should "keep sphere size constant when window height changes" in {
    assume(OptiXRenderer.isLibraryLoaded, "OptiX library not available")

    val (renderer, baselineVDiam, baselineHDiam) = createBaselineRender()

    try {
      val eye = Array(0f, 0f, 5f)
      val lookAt = Array(0f, 0f, 0f)
      val up = Array(0f, 1f, 0f)
      val fov = 45f

      // Render at 800x1600 (tall 1:2 - height doubled)
      renderer.updateImageDimensions(800, 1600)
      renderer.setCamera(eye, lookAt, up, fov)
      val img = renderer.render(800, 1600)
      TestUtilities.savePPM("window_resize_vertical_800x1600.ppm", img, 800, 1600)

      val vDiam = measureVerticalDiameter(img, 800, 1600)
      val hDiam = measureHorizontalDiameter(img, 800, 1600)
      logger.info(s"Vertical resize 800x1600: vertical=${vDiam}px, horizontal=${hDiam}px")

      // When height changes, horizontal diameter should stay the same
      Math.abs(hDiam - baselineHDiam) should be < 5

      // Sphere should still be circular
      Math.abs(vDiam - hDiam) should be < 5

    } finally {
      renderer.dispose()
    }
  }
}
