package menger

import scala.annotation.targetName

import menger.common.Const
import menger.objects.higher_d.Projection
import menger.objects.higher_d.Rotation

case class RotationProjectionParameters(
  rotXW: Float = 0, rotYW: Float = 0, rotZW: Float = 0,
  eyeW: Float = Const.defaultEyeW, screenW: Float = Const.defaultScreenW,
  rotX: Float = 0, rotY: Float = 0, rotZ: Float = 0
):
  lazy val projection: Projection = Projection(eyeW, screenW)
  lazy val rotation: Rotation = Rotation(this)

  @targetName("plus")
  def +(other: RotationProjectionParameters): RotationProjectionParameters =
    val proj = projection + other.projection
    RotationProjectionParameters(
      rotXW + other.rotXW, rotYW + other.rotYW, rotZW + other.rotZW,
      proj.eyeW, proj.screenW,
      rotX + other.rotX, rotY + other.rotY, rotZ + other.rotZ
    )

  private def hasXYZRotation: Boolean = rotX != 0 || rotY != 0 || rotZ != 0
  private def hasXYZWRotation: Boolean = rotXW != 0 || rotYW != 0 || rotZW != 0
  private def hasNonDefaultEyeW: Boolean = eyeW != Const.defaultEyeW
  private def hasNonDefaultScreenW: Boolean = screenW != Const.defaultScreenW

  override def toString: String = 
    "|" + 
      (if hasXYZRotation then s"x: $rotX y: $rotY z: $rotZ " else "") +
      (if hasXYZWRotation then s"xw: $rotXW yw: $rotYW zw: $rotZW " else "") +
      (if hasNonDefaultEyeW then s"eyeW=$eyeW " else "") +
      (if hasNonDefaultScreenW then s"screenW=$screenW " else "") + 
    "|"

object RotationProjectionParameters:
  def apply(opts: MengerCLIOptions): RotationProjectionParameters =
    RotationProjectionParameters(
      rotXW = opts.rotXW(), rotYW = opts.rotYW(), rotZW = opts.rotZW(),
      eyeW = opts.projectionEyeW(), screenW = opts.projectionScreenW(),
      rotX = opts.rotX(), rotY = opts.rotY(), rotZ = opts.rotZ()
    )

  def apply(rotXW: Float, rotYW: Float, rotZW: Float): RotationProjectionParameters =
    RotationProjectionParameters(rotXW, rotYW, rotZW, Const.defaultEyeW, Const.defaultScreenW)

