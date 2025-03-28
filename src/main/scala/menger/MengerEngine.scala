package menger

import com.badlogic.gdx.Game
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, ModelInstance}
import menger.objects.{Builder, Geometry, SpongeBySurface, SpongeByVolume}
import menger.objects.higher_d.{RotatedProjection, Tesseract, TesseractSponge, TesseractSponge2}

abstract class MengerEngine(
  val spongeType: String, val spongeLevel: Int,
  val rotationProjectionParameters: RotationProjectionParameters, val lines: Boolean
) extends Game:
  protected val material: Material = Builder.WHITE_MATERIAL
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
  ): Geometry =
    spongeType match
      case "square" => SpongeBySurface(level, material, primitiveType)
      case "cube" => SpongeByVolume(level, material, primitiveType)
      case "tesseract" => RotatedProjection(Tesseract(), currentRotProj, material, primitiveType)
      case "tesseract-sponge" => RotatedProjection(
        TesseractSponge(level), currentRotProj, material, primitiveType
      )
      case "tesseract-sponge-2" => RotatedProjection(
        TesseractSponge2(level), currentRotProj, material, primitiveType
      )
      case _ => throw new IllegalArgumentException(s"Unknown sponge type: $spongeType")
