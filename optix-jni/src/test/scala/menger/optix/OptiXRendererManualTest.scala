package menger.optix

import java.io.{File, FileOutputStream}

/**
 * Manual test utility for viewing OptiX rendering output.
 * Run with: sbt "project optixJni" "runMain menger.optix.OptiXRendererManualTest"
 *
 * This will create output.ppm which can be viewed with image viewers like GIMP,
 * ImageMagick, or converted with: convert output.ppm output.png
 */
object OptiXRendererManualTest {

  def main(args: Array[String]): Unit = {
    println("=== OptiX Manual Rendering Test ===")

    val width = 800
    val height = 600

    val renderer = new OptiXRenderer()

    println("Initializing OptiX...")
    if (!renderer.initialize()) {
      println("ERROR: Failed to initialize OptiX")
      System.exit(1)
    }

    println("Configuring scene...")
    // Sphere at origin
    renderer.setSphere(0.0f, 0.0f, 0.0f, 1.5f)

    // Camera looking at sphere from z=3
    val eye = Array(0.0f, 0.0f, 3.0f)
    val lookAt = Array(0.0f, 0.0f, 0.0f)
    val up = Array(0.0f, 1.0f, 0.0f)
    renderer.setCamera(eye, lookAt, up, 60.0f)

    // Light from upper right
    val lightDir = Array(0.5f, 0.5f, -0.5f)
    renderer.setLight(lightDir, 1.0f)

    println(s"Rendering ${width}x${height} image...")
    val imageData = renderer.render(width, height)

    println("Saving to output.ppm...")
    savePPM("output.ppm", imageData, width, height)

    println("Cleaning up...")
    renderer.dispose()

    println("Done! View the image with:")
    println("  - GIMP: gimp output.ppm")
    println("  - ImageMagick: display output.ppm")
    println("  - Convert to PNG: convert output.ppm output.png")
  }

  /**
   * Save image data as PPM (Portable Pixel Map) format.
   * PPM is a simple uncompressed image format that's easy to generate.
   */
  def savePPM(filename: String, data: Array[Byte], width: Int, height: Int): Unit = {
    val file = new File(filename)
    val out = new FileOutputStream(file)

    try {
      // PPM header
      val header = s"P6\n$width $height\n255\n"
      out.write(header.getBytes("ASCII"))

      // Write RGB data (skip alpha channel)
      for (i <- 0 until width * height) {
        val offset = i * 4
        // Convert signed bytes to unsigned
        out.write(data(offset) & 0xFF)     // R
        out.write(data(offset + 1) & 0xFF) // G
        out.write(data(offset + 2) & 0xFF) // B
        // Skip alpha (data(offset + 3))
      }

      println(s"Saved ${width}x${height} image to $filename (${file.length()} bytes)")
    } finally {
      out.close()
    }
  }
}
