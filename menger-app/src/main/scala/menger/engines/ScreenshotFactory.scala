package menger.engines

import java.io.File

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.files.FileHandle
import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.PixmapIO
import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.RenderHealth
import io.github.lene.optix.UniformRenderException

object ScreenshotFactory extends LazyLogging:

  def saveScreenshot(fileName: String, allowUniformRender: Boolean = false): Unit =
    val safePath = sanitizePath(fileName)
    val file     = new File(safePath)
    Option(file.getParentFile).foreach(_.mkdirs())
    val pixmap = getScreenshot(0, 0, Gdx.graphics.getWidth, Gdx.graphics.getHeight)
    try
      checkUniformity(pixmap, safePath, allowUniformRender)
      PixmapIO.writePNG(FileHandle(safePath), pixmap)
    finally pixmap.dispose()

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def checkUniformity(pixmap: Pixmap, path: String, allow: Boolean): Unit =
    val width  = pixmap.getWidth
    val height = pixmap.getHeight
    val rgba   = pixmapToRgbaBytes(pixmap)
    try
      RenderHealth.checkRgba(rgba, width, height)
    catch
      case e: UniformRenderException =>
        if allow then
          logger.warn(s"render health check: ${e.getMessage} (allowed for $path)")
        else
          val full = s"render health check failed for $path: ${e.getMessage}. " +
            "Pass --allow-uniform-render to permit uniform output."
          logger.error(full)
          throw UniformRenderException(full)

  private def pixmapToRgbaBytes(pixmap: Pixmap): Array[Byte] =
    val width  = pixmap.getWidth
    val height = pixmap.getHeight
    val out    = new Array[Byte](width * height * 4)
    (0 until height).foreach: y =>
      (0 until width).foreach: x =>
        val rgba8888 = pixmap.getPixel(x, y)
        val base     = (y * width + x) * 4
        out(base)     = ((rgba8888 >>> 24) & 0xFF).toByte
        out(base + 1) = ((rgba8888 >>> 16) & 0xFF).toByte
        out(base + 2) = ((rgba8888 >>> 8)  & 0xFF).toByte
        out(base + 3) = ( rgba8888         & 0xFF).toByte
    out

  private[menger] def sanitizePath(path: String): String =
    require(path.nonEmpty, "File path cannot be empty")
    require(!path.contains(".."), "Path traversal not allowed")

    val sanitized = path.filter(c =>
      c.isLetterOrDigit || c == '_' || c == '-' || c == '.' || c == '/'
    )

    require(sanitized.nonEmpty, "File path becomes empty after sanitization")
    if sanitized.toLowerCase.endsWith(".png") then sanitized else s"$sanitized.png"

  // OpenGL framebuffer has Y=0 at bottom, but PNG format has Y=0 at top.
  // LibGDX handles this internally for display, but we must flip when saving.
  private def getScreenshot(x: Int, y: Int, w: Int, h: Int): Pixmap =
    val original = Pixmap.createFromFrameBuffer(x, y, w, h)
    flipVertically(original)

  private def flipVertically(pixmap: Pixmap): Pixmap =
    val width = pixmap.getWidth
    val height = pixmap.getHeight
    val flipped = new Pixmap(width, height, pixmap.getFormat)
    (0 until height).foreach { y =>
      (0 until width).foreach { x =>
        flipped.drawPixel(x, height - 1 - y, pixmap.getPixel(x, y))
      }
    }
    pixmap.dispose()
    flipped
