package menger.engines

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

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
    if safePath.toLowerCase.endsWith(".pfm") then savePfm(safePath, allowUniformRender)
    else savePng(safePath, allowUniformRender)

  private def savePng(safePath: String, allowUniformRender: Boolean): Unit =
    val pixmap = getScreenshot(0, 0, Gdx.graphics.getWidth, Gdx.graphics.getHeight)
    try
      checkUniformity(pixmap, safePath, allowUniformRender)
      PixmapIO.writePNG(FileHandle(safePath), pixmap)
    finally pixmap.dispose()

  // Linear float dump for physically-based validation against external renderers
  // (pbrt). PFM rows are stored bottom-to-top, matching the OpenGL framebuffer, so
  // the raw (unflipped) framebuffer is used. Values are byte/255: the framebuffer is
  // already 8-bit, so this is quantized-linear clipped to [0,1] — adequate for
  // MSE/FLIP comparison but not true HDR (see docs/BACKLOG.md F-HDR-FILM).
  private def savePfm(safePath: String, allowUniformRender: Boolean): Unit =
    val width  = Gdx.graphics.getWidth
    val height = Gdx.graphics.getHeight
    val pixmap = Pixmap.createFromFrameBuffer(0, 0, width, height)
    try
      checkUniformity(pixmap, safePath, allowUniformRender)
      writePfm(safePath, pixmap, width, height)
    finally pixmap.dispose()

  private def writePfm(safePath: String, pixmap: Pixmap, width: Int, height: Int): Unit =
    val buffer = ByteBuffer.allocate(width * height * 3 * 4).order(ByteOrder.LITTLE_ENDIAN)
    (0 until height).foreach: y =>
      (0 until width).foreach: x =>
        val rgba8888 = pixmap.getPixel(x, y)
        val r = ((rgba8888 >>> 24) & 0xFF) / 255.0f
        val g = ((rgba8888 >>> 16) & 0xFF) / 255.0f
        val b = ((rgba8888 >>> 8)  & 0xFF) / 255.0f
        buffer.putFloat(r).putFloat(g).putFloat(b)
    val out = new FileOutputStream(safePath)
    try
      out.write(s"PF\n$width $height\n-1.0\n".getBytes("US-ASCII"))
      out.write(buffer.array())
    finally out.close()

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def checkUniformity(pixmap: Pixmap, path: String, allow: Boolean): Unit =
    val width  = pixmap.getWidth
    val height = pixmap.getHeight
    val rgba   = pixmapToRgbaBytes(pixmap)
    checkRgbaForSave(rgba, width, height, path, allow)

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private[menger] def checkRgbaForSave(
    rgba: Array[Byte],
    width: Int,
    height: Int,
    path: String,
    allow: Boolean
  ): Unit =
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
    val lower = sanitized.toLowerCase
    if lower.endsWith(".png") || lower.endsWith(".pfm") then sanitized else s"$sanitized.png"

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
