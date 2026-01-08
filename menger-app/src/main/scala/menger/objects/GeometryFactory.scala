package menger.objects

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.math.Vector3
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.common.UnknownGeometryException
import menger.objects.higher_d.FractionalRotatedProjection
import menger.objects.higher_d.RotatedProjection
import menger.objects.higher_d.Tesseract
import menger.objects.higher_d.TesseractSponge
import menger.objects.higher_d.TesseractSponge2

object GeometryFactory:

  def create(
    spongeType: String,
    level: Float,
    material: Material,
    primitiveType: Int,
    rotationProjection: RotationProjectionParameters
  )(using ProfilingConfig): Try[Geometry] =
    spongeType match
      case "square" => Try(Square(Vector3.Zero, 1f, material, primitiveType))
      case "cube" => Try(Cube(Vector3.Zero, 1f, material, primitiveType))
      case "square-sponge" => Try(SpongeBySurface(Vector3.Zero, 1f, level, material, primitiveType))
      case "cube-sponge" => Try(SpongeByVolume(Vector3.Zero, 1f, level, material, primitiveType))
      case "tesseract" => Try(RotatedProjection(
        Vector3.Zero, 1f, Tesseract(),
        rotationProjection.projection, rotationProjection.rotation,
        material, primitiveType
      ))
      case "tesseract-sponge" => Try(FractionalRotatedProjection(
        Vector3.Zero, 1f,
        (l: Float) => TesseractSponge(l),
        level, rotationProjection.projection, rotationProjection.rotation,
        material, primitiveType
      ))
      case "tesseract-sponge-2" => Try(FractionalRotatedProjection(
        Vector3.Zero, 1f,
        (l: Float) => TesseractSponge2(l),
        level, rotationProjection.projection, rotationProjection.rotation,
        material, primitiveType
      ))
      case "sphere" =>
        Failure(UnsupportedOperationException("Sphere rendering requires --optix flag"))
      case composite if composite.startsWith("composite[") =>
        Composite.parseCompositeFromCLIOption(
          composite, level, material, primitiveType, rotationProjection,
          create
        )
      case _ => Failure(UnknownGeometryException(spongeType, supportedTypes.toSeq.sorted))

  def createWithOverlay(
    spongeType: String,
    level: Float,
    faceColor: Option[com.badlogic.gdx.graphics.Color],
    lineColor: Option[com.badlogic.gdx.graphics.Color],
    defaultMaterial: Material,
    rotationProjection: RotationProjectionParameters
  )(using ProfilingConfig): Try[Geometry] =
    val isOverlayMode = faceColor.isDefined && lineColor.isDefined
    if !isOverlayMode then
      create(spongeType, level, defaultMaterial, GL20.GL_TRIANGLES, rotationProjection)
    else
      val faceMaterial = Builder.material(faceColor.get)
      val lineMaterial = Builder.material(lineColor.get)
      for
        faces <- create(spongeType, level, faceMaterial, GL20.GL_TRIANGLES, rotationProjection)
        lines <- create(spongeType, level, lineMaterial, GL20.GL_LINES, rotationProjection)
      yield Composite(Vector3.Zero, 1f, List(faces, lines))

  val supportedTypes: Set[String] = Set(
    "square", "cube", "square-sponge", "cube-sponge",
    "tesseract", "tesseract-sponge", "tesseract-sponge-2", "sphere"
  )

  def isValidType(spongeType: String): Boolean =
    supportedTypes.contains(spongeType) || spongeType.startsWith("composite[")
