package menger.objects

import scala.math.abs
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.attributes.{BlendingAttribute, ColorAttribute}
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3


class SpongeByVolume(
  center: Vector3 = Vector3.Zero, scale: Float = 1f,
  level: Float, material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Cube(center, scale, material, primitiveType):
  Option(material.get(ColorAttribute.Diffuse))
    .collect { case attr: ColorAttribute => attr }
    .foreach(attr => println(s"SpongeByVolume(level=$level) alpha: ${attr.color.a}"))
  private val subSponge = if level > 1
  then SpongeByVolume(Vector3.Zero, 1f, level - 1, material, primitiveType)
  else Cube(Vector3.Zero, 1f, material, primitiveType)
  private val transparentSponge: Option[SpongeByVolume] =
    if level != level.toInt then
      val newAlpha = Option(material.get(ColorAttribute.Diffuse))
        .collect { case attr: ColorAttribute => attr }
        .map(_.color.a * (level - level.floor))
        .getOrElse(1.0f)
      val newMaterial = createMaterialWithAlpha(material, newAlpha)
      Some(SpongeByVolume(center, scale, (level+1).floor, newMaterial, primitiveType))
    else None


  def createMaterialWithAlpha(originalMaterial: Material, newAlpha: Float): Material =
    val newMaterial = new Material(originalMaterial)

    Option(newMaterial.get(ColorAttribute.Diffuse))
      .collect { case attr: ColorAttribute => attr }
      .foreach(_.color.a = newAlpha)

    Option(newMaterial.get(ColorAttribute.Specular))
      .collect { case attr: ColorAttribute => attr }
      .foreach(_.color.a = newAlpha)

    newMaterial.set(new BlendingAttribute(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA, newAlpha))

    newMaterial

  override def getModel: List[ModelInstance] =
    if level <= 0 then super.getModel
    else if level == level.toInt then
      getIntegerModel
    else
      List(getIntegerModel, transparentSponge.map(_.getModel).getOrElse(Nil)).flatten


  private def getIntegerModel = {
    // Integer level - use current implementation
    logTime("at", 10) {
      val shift = scale / 3f
      val subCubeList = for (
        xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1 if abs(xx) + abs(yy) + abs(zz) > 1
      )
      yield SpongeByVolume(
        Vector3(center.x + xx * shift, center.y + yy * shift, center.z + zz * shift), scale / 3f,
        level - 1, material, primitiveType
      ).getModel

      subCubeList.flatten.toList
    }
  }

  override def toString: String =
    s"SpongeByVolume(level=${float2string(level)}, ${6 * Math.pow(20, level.toInt).toLong} faces)"
