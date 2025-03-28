package menger

import menger.objects.higher_d.{Projection, Rotation}

import scala.annotation.targetName

case class RotationProjectionParameters(
  rotXW: Float = 0, rotYW: Float = 0, rotZW: Float = 0,
  eyeW: Float = Const.defaultEyeW, screenW: Float = Const.defaultScreenW,
  rotX: Float = 0, rotY: Float = 0, rotZ: Float = 0
):
  lazy val projection: Projection = Projection(eyeW, screenW)
  lazy val rotation: Rotation = Rotation(rotXW, rotYW, rotZW)
  
  @targetName("plus")
  def +(other: RotationProjectionParameters): RotationProjectionParameters =
    val proj = projection + other.projection
    RotationProjectionParameters(
      rotXW + other.rotXW, rotYW + other.rotYW, rotZW + other.rotZW, proj.eyeW, proj.screenW
    )

object RotationProjectionParameters:
  def apply(opts: MengerCLIOptions): RotationProjectionParameters =
    RotationProjectionParameters(
      opts.rotXW(), opts.rotYW(), opts.rotZW(), opts.projectionEyeW(), opts.projectionScreenW()
    )

  def apply(rotXW: Float, rotYW: Float, rotZW: Float): RotationProjectionParameters =
    RotationProjectionParameters(rotXW, rotYW, rotZW, Const.defaultEyeW, Const.defaultScreenW)

