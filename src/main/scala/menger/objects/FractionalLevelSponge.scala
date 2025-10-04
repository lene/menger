package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.attributes.BlendingAttribute
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute
import com.badlogic.gdx.math.Vector3

trait FractionalLevelSponge:
  def center: Vector3
  def scale: Float
  def level: Float
  def material: Material
  def primitiveType: Int

  /** Factory method for creating instances of the specific sponge type */
  protected def createInstance(
    center: Vector3, scale: Float, level: Float, material: Material, primitiveType: Int
  ): Geometry & FractionalLevelSponge

  private[objects] lazy val transparentSponge: Option[Geometry & FractionalLevelSponge] =
    if level.isValidInt then None
    else Some(createInstance(center, scale, level.floor, transparentMaterial, primitiveType))

  private[objects] lazy val nextLevelSponge: Option[Geometry & FractionalLevelSponge] =
    if level.isValidInt then None
    else Some(createInstance(center, scale, (level+1).floor, material, primitiveType))

  private[objects] def transparentMaterial =
    val newAlpha = FractionalLevelSponge.calculateAlphaForFractionalLevel(material, level - level.floor)
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

object FractionalLevelSponge:
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
