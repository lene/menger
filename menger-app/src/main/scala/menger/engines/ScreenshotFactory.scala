package menger.engines

import java.io.File

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.files.FileHandle
import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.PixmapIO

object ScreenshotFactory:
  def saveScreenshot(fileName: String): Unit =
    val safePath = sanitizePath(fileName)
    val file = new File(safePath)
    Option(file.getParentFile).foreach(_.mkdirs())
    val pixmap = getScreenshot(0, 0, Gdx.graphics.getWidth, Gdx.graphics.getHeight)
    PixmapIO.writePNG(FileHandle(safePath), pixmap)
    pixmap.dispose()

  private[menger] def sanitizePath(path: String): String =
    require(path.nonEmpty, "File path cannot be empty")
    require(!path.contains(".."), "Path traversal not allowed")

    val sanitized = path.filter(c =>
      c.isLetterOrDigit || c == '_' || c == '-' || c == '.' || c == '/'
    )

    require(sanitized.nonEmpty, "File path becomes empty after sanitization")
    val withExtension =
      if sanitized.toLowerCase.endsWith(".png") then sanitized else s"$sanitized.png"
    if withExtension.startsWith("/") then withExtension.drop(1) else withExtension

  private[menger] def sanitizeFileName(fileName: String): String =
    require(fileName.nonEmpty, "File name cannot be empty")

    val sanitized = fileName.filter(c => c.isLetterOrDigit || c == '_' || c == '-' || c == '.')

    require(sanitized.nonEmpty, "File name becomes empty after sanitization")
    if sanitized.toLowerCase.endsWith(".png") then sanitized else s"$sanitized.png"

  private def getScreenshot(x: Int, y: Int, w: Int, h: Int) =
    Pixmap.createFromFrameBuffer(x, y, w, h)
