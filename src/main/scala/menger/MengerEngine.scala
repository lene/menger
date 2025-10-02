package menger

import scala.util.Try

import com.badlogic.gdx.Game
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.objects.Builder
import menger.objects.Composite
import menger.objects.Cube
import menger.objects.Geometry
import menger.objects.SpongeBySurface
import menger.objects.SpongeByVolume
import menger.objects.Square
import menger.objects.higher_d.RotatedProjection
import menger.objects.higher_d.Tesseract
import menger.objects.higher_d.TesseractSponge
import menger.objects.higher_d.TesseractSponge2

abstract class MengerEngine(
  val spongeType: String, val spongeLevel: Float,
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
    spongeType: String, level: Float, material: Material, primitiveType: Int
  ): Try[Geometry] = {
    MengerEngine.count += 1
    println(s"Generating object #${MengerEngine.count}: type='$spongeType', level=$level")
    spongeType match
      case "square" => Try(Square(Vector3.Zero, 1f, material, primitiveType))
      case "cube" => Try(Cube(Vector3.Zero, 1f, material, primitiveType))
      case "square-sponge" => Try(SpongeBySurface(Vector3.Zero, 1f, level.toInt, material, primitiveType))
      case "cube-sponge" => Try(SpongeByVolume(Vector3.Zero, 1f, level, material, primitiveType))
      case "tesseract" => Try(RotatedProjection(Vector3.Zero, 1f, Tesseract(), currentRotProj.projection, currentRotProj.rotation, material, primitiveType))
      case "tesseract-sponge" => Try(RotatedProjection(
        Vector3.Zero, 1f, TesseractSponge(level.toInt), currentRotProj.projection, currentRotProj.rotation, material, primitiveType
      ))
      case "tesseract-sponge-2" => Try(RotatedProjection(
        Vector3.Zero, 1f, TesseractSponge2(level.toInt), currentRotProj.projection, currentRotProj.rotation, material, primitiveType
      ))
      case composite if composite.startsWith("composite[") =>
        Composite.parseCompositeFromCLIOption(composite, level, material, primitiveType, generateObject)
      case _ => scala.util.Failure(IllegalArgumentException(s"Unknown sponge type: $spongeType"))
  }

object MengerEngine:
  var count: Int = 0
