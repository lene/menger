package menger

import scala.util.Try

import com.badlogic.gdx.Game
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import menger.objects.Builder
import menger.objects.Geometry
import menger.objects.SpongeBySurface
import menger.objects.SpongeByVolume
import menger.objects.higher_d.RotatedProjection
import menger.objects.higher_d.Tesseract
import menger.objects.higher_d.TesseractSponge
import menger.objects.higher_d.TesseractSponge2

abstract class MengerEngine(
  val spongeType: String, val spongeLevel: Int,
  val rotationProjectionParameters: RotationProjectionParameters, val lines: Boolean, val color: Color
) extends Game:
  protected val material: Material = Builder.material(color)
  protected lazy val primitiveType: Int = if lines then GL20.GL_LINES else GL20.GL_TRIANGLES
  protected def gdxResources: GDXResources
  protected def drawables: List[ModelInstance]
  override def resume(): Unit = {}
  override def pause(): Unit = {}
  override def dispose(): Unit = gdxResources.dispose()
  override def resize(width: Int, height: Int): Unit = gdxResources.resize()
  def currentRotProj: RotationProjectionParameters = rotationProjectionParameters

  protected def generateObject(
    spongeType: String, level: Int, material: Material, primitiveType: Int
  ): Try[Geometry] =
    spongeType match
      case "square" => Try(SpongeBySurface(level, material, primitiveType))
      case "cube" => Try(SpongeByVolume(level, material, primitiveType))
      case "tesseract" => Try(RotatedProjection(Tesseract(), currentRotProj, material, primitiveType))
      case "tesseract-sponge" => Try(RotatedProjection(
        TesseractSponge(level), currentRotProj, material, primitiveType
      ))
      case "tesseract-sponge-2" => Try(RotatedProjection(
        TesseractSponge2(level), currentRotProj, material, primitiveType
      ))
      case _ => scala.util.Failure(IllegalArgumentException(s"Unknown sponge type: $spongeType"))
