package menger.spec

import menger.common.Color

case class ObjectRotation(x: Radians = Radians.Zero, y: Radians = Radians.Zero, z: Radians = Radians.Zero)

case class ConeGeometry(
  apex: Option[(Float, Float, Float)] = None,
  base: Option[(Float, Float, Float)] = None,
  radius: Option[Float] = None
)

case class PlaneGeometry(
  normal: Option[(Float, Float, Float)] = None,
  distance: Option[Float] = None,
  color2: Option[Color] = None,
  checkerSize: Float = 1.0f
)

case class ProceduralSpec(proceduralType: Int = 0, scale: Float = 1.0f)

case class CurveData(points: Vector[Float], widths: Vector[Float])

case class TextureMaps(normalMap: Option[String] = None, roughnessMap: Option[String] = None)
