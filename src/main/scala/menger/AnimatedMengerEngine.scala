package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.files.FileHandle
import com.badlogic.gdx.graphics.{Color, Pixmap, PixmapIO}
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3

class AnimatedMengerEngine(
  spongeType: String, spongeLevel: Int,
  rotationProjectionParameters: RotationProjectionParameters = RotationProjectionParameters(),
  lines: Boolean, color: Color, val animationSpecifications: AnimationSpecifications, 
  val saveName: Option[String]
) extends MengerEngine(spongeType, spongeLevel, rotationProjectionParameters, lines, color):
  private var currentFrame: Int = 0

  protected def drawables: List[ModelInstance] = 
    generateObject(spongeType, spongeLevel, material, primitiveType).at(Vector3(0, 0, 0), 1)

  protected def gdxResources: GDXResources = GDXResources(None)

  override def currentRotProj: RotationProjectionParameters =
    val r = rotationProjectionParameters + animationSpecifications.rotationProjectionParameters(currentFrame)
    Gdx.app.log(s"${getClass.getSimpleName}", s"frame: $currentFrame $r ${currentSaveName.getOrElse("")}")
    r

  override def create(): Unit =
    Gdx.app.log(s"${getClass.getSimpleName}", s"Animating for $animationSpecifications")

  override def render(): Unit =
    super.render()
    gdxResources.render(drawables)
    saveImage()
    nextStep()

  private def nextStep(): Unit = 
    currentFrame += 1
    if currentFrame >= animationSpecifications.numFrames then
      Gdx.app.exit()

  private def currentSaveName: Option[String] = saveName.map(_.format(currentFrame))

  private def saveImage():  Unit = currentSaveName.foreach(ScreenshotFactory.saveScreenshot)


object ScreenshotFactory:
  def saveScreenshot(fileName: String): Unit =
    val pixmap = getScreenshot(0, 0, Gdx.graphics.getWidth, Gdx.graphics.getHeight)
    PixmapIO.writePNG(FileHandle(fileName), pixmap)
    pixmap.dispose()

  private def getScreenshot(x: Int, y: Int, w: Int, h: Int) =
    Pixmap.createFromFrameBuffer(x, y, w, h)
