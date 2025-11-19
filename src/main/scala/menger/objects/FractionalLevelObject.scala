package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.attributes.BlendingAttribute
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute


trait FractionalLevelObject:
  def level: Float
  def material: Material
  
  
  protected[objects] def calculateAlpha(): Float = {
    val fractionalPart = level - level.floor
    Option(material.get(ColorAttribute.Diffuse))
      .collect { case attr: ColorAttribute => attr }
      .map(_.color.a * (1.0f - fractionalPart))
      .getOrElse(1.0f)
  }

  protected[objects] def transparentMaterial: Material =
    materialWithAlpha(calculateAlpha())

  def materialWithAlpha(alpha: Float): Material =
    val newMaterial = new Material(material)

    Option(newMaterial.get(ColorAttribute.Diffuse))
      .collect { case attr: ColorAttribute => attr }
      .foreach(_.color.a = alpha)
    Option(newMaterial.get(ColorAttribute.Specular))
      .collect { case attr: ColorAttribute => attr }
      .foreach(_.color.a = alpha)
    newMaterial.set(new BlendingAttribute(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA, alpha))

    newMaterial
