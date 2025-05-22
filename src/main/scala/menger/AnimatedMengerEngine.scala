package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.files.FileHandle
import com.badlogic.gdx.graphics.PixmapIO
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import com.badlogic.gdx.utils.ScreenUtils

class AnimatedMengerEngine(
  spongeType: String, spongeLevel: Int,
  rotationProjectionParameters: RotationProjectionParameters = RotationProjectionParameters(),
  lines: Boolean, val animationSpecifications: AnimationSpecifications, val saveName: String
) extends MengerEngine(spongeType, spongeLevel, rotationProjectionParameters, lines):
  private var currentFrame: Int = 0

  protected def drawables: List[ModelInstance] = 
    generateObject(spongeType, spongeLevel, material, primitiveType).at(Vector3(0, 0, 0), 1)

  protected def gdxResources: GDXResources = GDXResources(None)

  override def currentRotProj: RotationProjectionParameters =
    val r = rotationProjectionParameters + animationSpecifications.rotationProjectionParameters(currentFrame)
    Gdx.app.log(s"${getClass.getSimpleName}.currentRotProj()", s"frame: $currentFrame $r $currentSaveName")
    r

  override def create(): Unit =
    Gdx.app.log(
      s"${getClass.getSimpleName}.create()",
      s"Animating for $animationSpecifications ${if saveName.nonEmpty then s"and saving to $saveName" else ""}"
    )

  override def render(): Unit =
    super.render()
    gdxResources.render(drawables)
    saveImage()
    nextStep()
    

  private def nextStep(): Unit = 
    currentFrame += 1
    if currentFrame >= animationSpecifications.numFrames then
      Gdx.app.exit()

  private def currentSaveName: Option[String] =
    if saveName.nonEmpty then Some(saveName.format(currentFrame)) else None

  private def saveImage():  Unit =
    if saveName.nonEmpty then
      val fileName = currentSaveName.getOrElse("")
      Gdx.app.log(s"${getClass.getSimpleName}.saveImage()", s"Saving image to $fileName")
      ScreenshotFactory.saveScreenshot(fileName)
    else
      Gdx.app.log(s"${getClass.getSimpleName}.saveImage()", "No save name provided, skipping image save.")


object ScreenshotFactory {
  private var counter = 1

  def saveScreenshot(fileName: String): Unit = {
      val fh = new FileHandle(fileName)
      val pixmap = getScreenshot(0, 0, Gdx.graphics.getWidth, Gdx.graphics.getHeight, false)
      PixmapIO.writePNG(fh, pixmap)
      pixmap.dispose
  }

  private def getScreenshot(x: Int, y: Int, w: Int, h: Int, yDown: Boolean) = {
    val pixmap = ScreenUtils.getFrameBufferPixmap(x, y, w, h)
    if (yDown) {
      // Flip the pixmap upside down
      val pixels = pixmap.getPixels
      val numBytes = w * h * 4
      val lines = new Array[Byte](numBytes)
      val numBytesPerLine = w * 4
      for (i <- 0 until h) {
        pixels.position((h - i - 1) * numBytesPerLine)
        pixels.get(lines, i * numBytesPerLine, numBytesPerLine)
      }
      pixels.clear
      pixels.put(lines)
    }
    pixmap
  }
}