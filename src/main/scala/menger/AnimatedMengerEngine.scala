package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.files.FileHandle
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.PixmapIO
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.typesafe.scalalogging.LazyLogging

class AnimatedMengerEngine(
  spongeType: String, spongeLevel: Float,
  rotationProjectionParameters: RotationProjectionParameters = RotationProjectionParameters(),
  lines: Boolean, color: Color, val animationSpecifications: AnimationSpecifications,
  val saveName: Option[String], faceColor: Option[Color] = None, lineColor: Option[Color] = None
)(using config: ProfilingConfig) extends MengerEngine(spongeType, spongeLevel, rotationProjectionParameters, lines, color, faceColor, lineColor)
with LazyLogging:
  private val frameCounter = java.util.concurrent.atomic.AtomicInteger(0)

  protected def drawables: List[ModelInstance] =
    given ProfilingConfig = profilingConfig
    val currentLevel = currentAnimatedLevel
    generateObjectWithOverlay(spongeType, currentLevel) match
      case scala.util.Success(geometry) => geometry.getModel
      case scala.util.Failure(exception) =>
        logger.error(s"Failed to create sponge type '$spongeType': ${exception.getMessage}")
        sys.exit(1)

  private def currentAnimatedLevel: Float =
    val currentFrame = frameCounter.get()
    animationSpecifications.level(currentFrame).getOrElse(spongeLevel)

  protected def gdxResources: GDXResources = GDXResources(None)

  override def currentRotProj: RotationProjectionParameters =
    val currentFrame = frameCounter.get()
    animationSpecifications.rotationProjectionParameters(currentFrame) match
      case scala.util.Success(animParams) =>
        rotationProjectionParameters + animParams
      case scala.util.Failure(exception) =>
        logger.error(s"Animation frame $currentFrame failed: ${exception.getMessage}")
        sys.exit(1)

  override def create(): Unit =
    logger.info(s"Animating for $animationSpecifications")

  override def render(): Unit =
    val currentFrame = frameCounter.get()
    logger.info(s"frame: $currentFrame |$currentRotProj| ${currentSaveName.getOrElse("")}")
    super.render()
    gdxResources.render(drawables)
    saveImage()
    nextStep()

  private def nextStep(): Unit =
    val nextFrame = frameCounter.incrementAndGet()
    if nextFrame >= animationSpecifications.numFrames then
      Gdx.app.exit()

  private def currentSaveName: Option[String] = saveName.map(_.format(frameCounter.get()))

  private def saveImage():  Unit = currentSaveName.foreach(ScreenshotFactory.saveScreenshot)


object ScreenshotFactory:
  def saveScreenshot(fileName: String): Unit =
    val safePath = sanitizeFileName(fileName)
    val pixmap = getScreenshot(0, 0, Gdx.graphics.getWidth, Gdx.graphics.getHeight)
    PixmapIO.writePNG(FileHandle(safePath), pixmap)
    pixmap.dispose()

  private[menger] def sanitizeFileName(fileName: String): String =
    require(fileName.nonEmpty, "File name cannot be empty")

    val sanitized = fileName.filter(c => c.isLetterOrDigit || c == '_' || c == '-' || c == '.')

    require(sanitized.nonEmpty, "File name becomes empty after sanitization")
    if sanitized.toLowerCase.endsWith(".png") then sanitized else s"$sanitized.png"

  private def getScreenshot(x: Int, y: Int, w: Int, h: Int) =
    Pixmap.createFromFrameBuffer(x, y, w, h)
