package menger.engines

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Game
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.GDXResources
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.objects.Builder
import menger.objects.Composite
import menger.objects.Cube
import menger.objects.Geometry
import menger.objects.SpongeBySurface
import menger.objects.SpongeByVolume
import menger.objects.Square
import menger.objects.higher_d.FractionalRotatedProjection
import menger.objects.higher_d.RotatedProjection
import menger.objects.higher_d.Tesseract
import menger.objects.higher_d.TesseractSponge
import menger.objects.higher_d.TesseractSponge2

abstract class MengerEngine(
  val spongeType: String, val spongeLevel: Float,
  val rotationProjectionParameters: RotationProjectionParameters, val lines: Boolean, val color: Color,
  val faceColor: Option[Color] = None, val lineColor: Option[Color] = None,
  val fpsLogIntervalMs: Int = 1000
)(using val profilingConfig: ProfilingConfig) extends Game:
  protected val material: Material = Builder.material(color)
  protected lazy val primitiveType: Int = if lines then GL20.GL_LINES else GL20.GL_TRIANGLES
  protected val isOverlayMode: Boolean = faceColor.isDefined && lineColor.isDefined
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
    given ProfilingConfig = profilingConfig
    spongeType match
      case "square" => Try(Square(Vector3.Zero, 1f, material, primitiveType))
      case "cube" => Try(Cube(Vector3.Zero, 1f, material, primitiveType))
      case "square-sponge" => Try(SpongeBySurface(Vector3.Zero, 1f, level, material, primitiveType))
      case "cube-sponge" => Try(SpongeByVolume(Vector3.Zero, 1f, level, material, primitiveType))
      case "tesseract" => Try(RotatedProjection(Vector3.Zero, 1f, Tesseract(), currentRotProj.projection, currentRotProj.rotation, material, primitiveType))
      case "tesseract-sponge" => Try(FractionalRotatedProjection(
        Vector3.Zero, 1f, (l: Float) => TesseractSponge(l), level, currentRotProj.projection, currentRotProj.rotation, material, primitiveType
      ))
      case "tesseract-sponge-2" => Try(FractionalRotatedProjection(
        Vector3.Zero, 1f, (l: Float) => TesseractSponge2(l), level, currentRotProj.projection, currentRotProj.rotation, material, primitiveType
      ))
      case "sphere" =>
        // NOTE: OptiXEngine doesn't use this - OptiX renders directly to 2D image
        // This case is for potential future CPU-based sphere rendering
        Failure(UnsupportedOperationException("Sphere rendering requires --optix flag"))
      case composite if composite.startsWith("composite[") =>
        Composite.parseCompositeFromCLIOption(composite, level, material, primitiveType, generateObject)
      case _ => Failure(IllegalArgumentException(s"Unknown sponge type: $spongeType"))
  }

  protected def generateObjectWithOverlay(spongeType: String, level: Float): Try[Geometry] =
    if !isOverlayMode then
      generateObject(spongeType, level, material, primitiveType)
    else
      val faceMaterial = Builder.material(faceColor.get)
      val lineMaterial = Builder.material(lineColor.get)
      for
        faces <- generateObject(spongeType, level, faceMaterial, GL20.GL_TRIANGLES)
        lines <- generateObject(spongeType, level, lineMaterial, GL20.GL_LINES)
      yield Composite(Vector3.Zero, 1f, List(faces, lines))
