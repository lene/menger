package menger.objects.higher_d

import scala.annotation.targetName

import menger.common.Const

/** A change of the 4D view: rotations, plus an eye-w when the event or animation sets one.
  * `eyeW` is `None` for a rotation-only change, which leaves the eye-w alone (menger#110: it
  * used to be the default value, which a user could then never set on purpose). */
case class RotationProjectionParameters(
  rotXW: Float = 0, rotYW: Float = 0, rotZW: Float = 0,
  eyeW: Option[Float] = None, screenW: Float = Const.defaultScreenW,
  rotX: Float = 0, rotY: Float = 0, rotZ: Float = 0
):
  lazy val projection: Projection = Projection(eyeW.getOrElse(Const.defaultEyeW), screenW)
  lazy val rotation: Rotation = Rotation(this)

  @targetName("plus")
  def +(other: RotationProjectionParameters): RotationProjectionParameters =
    val proj = projection + other.projection
    RotationProjectionParameters(
      rotXW + other.rotXW, rotYW + other.rotYW, rotZW + other.rotZW,
      Option.when(eyeW.isDefined || other.eyeW.isDefined)(proj.eyeW), proj.screenW,
      rotX + other.rotX, rotY + other.rotY, rotZ + other.rotZ
    )

  private def hasXYZRotation: Boolean = rotX != 0 || rotY != 0 || rotZ != 0
  private def hasXYZWRotation: Boolean = rotXW != 0 || rotYW != 0 || rotZW != 0
  private def hasNonDefaultScreenW: Boolean = screenW != Const.defaultScreenW

  override def toString: String =
    "|" +
      (if hasXYZRotation then s"x: $rotX y: $rotY z: $rotZ " else "") +
      (if hasXYZWRotation then s"xw: $rotXW yw: $rotYW zw: $rotZW " else "") +
      eyeW.fold("")(e => s"eyeW=$e ") +
      (if hasNonDefaultScreenW then s"screenW=$screenW " else "") +
    "|"

