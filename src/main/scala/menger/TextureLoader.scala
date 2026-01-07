package menger

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.nio.file.Paths
import javax.imageio.ImageIO

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import com.typesafe.scalalogging.LazyLogging

case class TextureData(
  name: String,
  data: Array[Byte],
  width: Int,
  height: Int
)

object TextureLoader extends LazyLogging:

  def load(filename: String, baseDir: String): Try[TextureData] =
    val path = resolvePath(filename, baseDir)
    loadFromPath(path, filename)

  def loadFromPath(path: Path, name: String): Try[TextureData] =
    validateFile(path).flatMap { file =>
      readImage(file, path).flatMap { image =>
        Try {
          val width = image.getWidth
          val height = image.getHeight
          val data = extractRgbaData(image)
          logger.debug(s"Loaded texture '$name' from $path (${width}x${height})")
          TextureData(name, data, width, height)
        }
      }
    }.recoverWith { case e: Exception =>
      logger.error(s"Failed to load texture '$name': ${e.getMessage}")
      Failure(e)
    }

  private def validateFile(path: Path): Try[java.io.File] =
    val file = path.toFile
    if !file.exists() then
      Failure(new java.io.FileNotFoundException(s"Texture file not found: $path"))
    else if !file.canRead then
      Failure(new java.io.IOException(s"Cannot read texture file: $path"))
    else
      Success(file)

  private def readImage(file: java.io.File, path: Path): Try[BufferedImage] =
    Try(ImageIO.read(file)).flatMap { image =>
      Option(image) match
        case Some(img) => Success(img)
        case None => Failure(new java.io.IOException(
          s"Failed to decode image: $path (unsupported format?)"))
    }

  private def resolvePath(filename: String, baseDir: String): Path =
    val filePath = Paths.get(filename)
    if filePath.isAbsolute then
      filePath
    else
      Paths.get(baseDir).resolve(filename)

  private def extractRgbaData(image: BufferedImage): Array[Byte] =
    val width = image.getWidth
    val height = image.getHeight
    val data = new Array[Byte](width * height * 4)

    for
      y <- 0 until height
      x <- 0 until width
    do
      val pixel = image.getRGB(x, y)
      val idx = (y * width + x) * 4
      data(idx) = ((pixel >> 16) & 0xFF).toByte     // R
      data(idx + 1) = ((pixel >> 8) & 0xFF).toByte  // G
      data(idx + 2) = (pixel & 0xFF).toByte         // B
      data(idx + 3) = ((pixel >> 24) & 0xFF).toByte // A

    data

  def createCheckerTexture(
    width: Int,
    height: Int,
    cellSize: Int = 8,
    color1: (Int, Int, Int) = (255, 255, 255),
    color2: (Int, Int, Int) = (0, 0, 0)
  ): TextureData =
    val data = new Array[Byte](width * height * 4)

    for
      y <- 0 until height
      x <- 0 until width
    do
      val idx = (y * width + x) * 4
      val isColor1 = ((x / cellSize) + (y / cellSize)) % 2 == 0
      val (r, g, b) = if isColor1 then color1 else color2
      data(idx) = r.toByte
      data(idx + 1) = g.toByte
      data(idx + 2) = b.toByte
      data(idx + 3) = 255.toByte // fully opaque

    TextureData("checker", data, width, height)

  def createGradientTexture(width: Int, height: Int): TextureData =
    val data = new Array[Byte](width * height * 4)

    for
      y <- 0 until height
      x <- 0 until width
    do
      val idx = (y * width + x) * 4
      data(idx) = ((x * 255) / width).toByte       // R: gradient left-right
      data(idx + 1) = ((y * 255) / height).toByte  // G: gradient top-bottom
      data(idx + 2) = 128.toByte                   // B: constant
      data(idx + 3) = 255.toByte                   // A: fully opaque

    TextureData("gradient", data, width, height)

  def createSolidTexture(
    width: Int,
    height: Int,
    r: Int,
    g: Int,
    b: Int,
    a: Int = 255
  ): TextureData =
    val data = new Array[Byte](width * height * 4)

    for
      y <- 0 until height
      x <- 0 until width
    do
      val idx = (y * width + x) * 4
      data(idx) = r.toByte
      data(idx + 1) = g.toByte
      data(idx + 2) = b.toByte
      data(idx + 3) = a.toByte

    TextureData("solid", data, width, height)
