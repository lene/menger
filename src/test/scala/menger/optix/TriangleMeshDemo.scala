package menger.optix

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import com.badlogic.gdx.math.Vector3
import menger.common.Color
import menger.common.ImageSize
import menger.common.TriangleMeshData
import menger.common.Vector
import menger.objects.Cube

object TriangleMeshDemo:

  private def savePNG(filename: String, data: Array[Byte], width: Int, height: Int): Unit =
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    (0 until width * height).foreach { i =>
      val offset = i * 4
      val r = data(offset) & 0xFF
      val g = data(offset + 1) & 0xFF
      val b = data(offset + 2) & 0xFF
      val a = data(offset + 3) & 0xFF
      val argb = (a << 24) | (r << 16) | (g << 8) | b
      val x = i % width
      val y = i / width
      image.setRGB(x, y, argb)
    }
    ImageIO.write(image, "PNG", new File(filename))

  // Simple triangle facing camera
  private def singleTriangle: TriangleMeshData =
    val vertices = Array[Float](
      -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f,
      0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f,
      0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f
    )
    TriangleMeshData(vertices, Array(0, 1, 2))

  // Quad (two triangles)
  private def quad: TriangleMeshData =
    val vertices = Array[Float](
      -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f,
      0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f,
      0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f,
      -0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f
    )
    TriangleMeshData(vertices, Array(0, 1, 2, 0, 2, 3))

  // Simple cube using the shared Cube.toTriangleMesh
  private def cube: TriangleMeshData =
    Cube(center = Vector3.Zero, scale = 0.8f).toTriangleMesh

  def main(args: Array[String]): Unit =
    println("Triangle Mesh Demo - Rendering to PNG files...")

    // Ensure library is loaded (trigger companion object initialization)
    require(OptiXRenderer.isLibraryLoaded, "OptiX native library failed to load")

    val renderer = new OptiXRenderer()
    renderer.initialize()

    val size = ImageSize(800, 600)

    // Set up camera
    renderer.setCamera(
      Vector[3](0.0f, 0.5f, 2.5f),
      Vector[3](0.0f, 0.0f, 0.0f),
      Vector[3](0.0f, 1.0f, 0.0f),
      60.0f
    )
    renderer.setLight(Vector[3](0.5f, 0.5f, -0.5f), 1.0f)

    // 1. Render opaque green triangle
    println("Rendering opaque triangle...")
    renderer.setTriangleMesh(singleTriangle)
    renderer.setTriangleMeshColor(Color(0.2f, 0.8f, 0.2f))
    val triangleImg = renderer.render(size).get
    savePNG("/tmp/triangle_opaque.png", triangleImg, size.width, size.height)
    println("  -> /tmp/triangle_opaque.png")

    // 2. Render transparent glass triangle
    println("Rendering transparent triangle...")
    renderer.setTriangleMeshColor(Color(0.9f, 0.9f, 1.0f, 0.3f))
    renderer.setTriangleMeshIOR(1.5f)
    val glassTriImg = renderer.render(size).get
    savePNG("/tmp/triangle_glass.png", glassTriImg, size.width, size.height)
    println("  -> /tmp/triangle_glass.png")

    // 3. Render opaque red quad
    println("Rendering quad...")
    renderer.setTriangleMesh(quad)
    renderer.setTriangleMeshColor(Color(0.8f, 0.2f, 0.2f))
    val quadImg = renderer.render(size).get
    savePNG("/tmp/quad_opaque.png", quadImg, size.width, size.height)
    println("  -> /tmp/quad_opaque.png")

    // 4. Render opaque blue cube
    println("Rendering cube...")
    renderer.setTriangleMesh(cube)
    renderer.setTriangleMeshColor(Color(0.2f, 0.4f, 0.8f))
    val cubeImg = renderer.render(size).get
    savePNG("/tmp/cube_opaque.png", cubeImg, size.width, size.height)
    println("  -> /tmp/cube_opaque.png")

    // 5. Render transparent glass cube
    println("Rendering glass cube...")
    renderer.setTriangleMeshColor(Color(0.95f, 0.95f, 1.0f, 0.2f))
    renderer.setTriangleMeshIOR(1.5f)
    val glassCubeImg = renderer.render(size).get
    savePNG("/tmp/cube_glass.png", glassCubeImg, size.width, size.height)
    println("  -> /tmp/cube_glass.png")

    renderer.dispose()
    println("\nDone! Check /tmp/*.png for output images.")
