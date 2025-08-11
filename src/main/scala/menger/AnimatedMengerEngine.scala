package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.files.FileHandle
import com.badlogic.gdx.graphics.{Pixmap, PixmapIO}
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3

import java.util.zip.Deflater

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

  private def saveImage(): Unit =
    if saveName.nonEmpty then
      val fileName = currentSaveName.getOrElse("")
      Gdx.app.log(s"${getClass.getSimpleName}.saveImage()", s"Saving image to $fileName")
      ScreenshotFactory.saveScreenshot(fileName)
    else
      Gdx.app.log(s"${getClass.getSimpleName}.saveImage()", "No save name provided, skipping image save.")


object ScreenshotFactory:
  def saveScreenshot(fileName: String): Unit =
      val pixmap = Pixmap.createFromFrameBuffer(0, 0, Gdx.graphics.getWidth, Gdx.graphics.getHeight)
      PixmapIO.writePNG(Gdx.files.external(fileName), pixmap, Deflater.BEST_COMPRESSION, true)
      pixmap.dispose()
