package menger.objects

import scala.math.abs

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.graphics.g3d.attributes.BlendingAttribute
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute
import com.badlogic.gdx.math.Vector3


class SpongeByVolume(
  center: Vector3 = Vector3.Zero, scale: Float = 1f,
  level: Float, material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Cube(center, scale, material, primitiveType):

  private val subSponge = if level > 1
  then SpongeByVolume(Vector3.Zero, 1f, level - 1, material, primitiveType)
  else Cube(Vector3.Zero, 1f, material, primitiveType)

  private[objects] lazy val transparentSponge: Option[SpongeByVolume] =
    if level.isValidInt then None
    else Some(SpongeByVolume(center, scale, level.floor, transparentMaterial, primitiveType))

  private[objects] lazy val nextLevelSponge: Option[SpongeByVolume] =
    if level.isValidInt then None
    else Some(SpongeByVolume(center, scale, (level+1).floor, material, primitiveType))

  private[objects] def transparentMaterial =
    val newAlpha = SpongeByVolume.calculateAlphaForFractionalLevel(material, level - level.floor)
    createMaterialWithAlpha(material, newAlpha)

  private[objects] def createMaterialWithAlpha(
    originalMaterial: Material, newAlpha: Float
  ): Material =
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
    logTime(s"getModel($level)", 5) {
      if level <= 0 then super.getModel
      else if level.isValidInt then getIntegerModel
      else List(
          nextLevelSponge.map(_.getModel).getOrElse(Nil),
          transparentSponge.map(_.getModel).getOrElse(Nil)
        ).flatten
    }

  private lazy val getIntegerModel =
    val shift = scale / 3f
    val subCubeList = for (
      xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1 if abs(xx) + abs(yy) + abs(zz) > 1
    )
    yield SpongeByVolume(
      Vector3(center.x + xx * shift, center.y + yy * shift, center.z + zz * shift), scale / 3f,
      level - 1, material, primitiveType
    ).getModel

    subCubeList.flatten.toList

  override def toString: String =
    s"SpongeByVolume(level=${float2string(level)}, ${6 * Math.pow(20, level.toInt).toLong} faces)"

object SpongeByVolume:

  /** Calculate alpha based on fractional part of level:
    * - When fractional part = 0: alpha = attr.color.a (material's full opacity)
    * - When fractional part → 1: alpha → 0 (fully transparent)
    *
    * @param material the material to extract the original alpha from
    * @param fractionalPart the fractional part of the level (0 to 1)
    * @return the calculated alpha value
    */
  def calculateAlphaForFractionalLevel(material: Material, fractionalPart: Float): Float =
    Option(material.get(ColorAttribute.Diffuse))
      .collect { case attr: ColorAttribute => attr }
      .map(_.color.a * (1.0f - fractionalPart))
      .getOrElse(1.0f)
