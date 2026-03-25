package menger.cli

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3

enum Axis:
  case X, Y, Z

case class PlaneSpec(axis: Axis, positive: Boolean, value: Float)

enum LightType:
  case DIRECTIONAL, POINT, AREA

enum AreaLightShape:
  case DISK

case class LightSpec(
  lightType: LightType,
  position: Vector3,
  intensity: Float,
  color: Color,
  normal: Vector3 = new Vector3(0f, -1f, 0f),
  radius: Float = 1.0f,
  shape: AreaLightShape = AreaLightShape.DISK,
  shadowSamples: Int = 4
)

object LightSpec:
  /** Convert common.Light to CLI LightSpec for rendering. */
  def fromCommonLight(light: menger.common.Light): LightSpec =
    light match
      case menger.common.Light.Directional(direction, clr, intensity) =>
        LightSpec(
          LightType.DIRECTIONAL,
          new Vector3(direction(0), direction(1), direction(2)),
          intensity,
          new Color(clr.r, clr.g, clr.b, clr.a)
        )
      case menger.common.Light.Point(position, clr, intensity) =>
        LightSpec(
          LightType.POINT,
          new Vector3(position(0), position(1), position(2)),
          intensity,
          new Color(clr.r, clr.g, clr.b, clr.a)
        )
      case menger.common.Light.Area(position, normal, radius, shape, clr, intensity, samples) =>
        LightSpec(
          LightType.AREA,
          new Vector3(position(0), position(1), position(2)),
          intensity,
          new Color(clr.r, clr.g, clr.b, clr.a),
          normal = new Vector3(normal(0), normal(1), normal(2)),
          radius = radius,
          shape = AreaLightShape.DISK,
          shadowSamples = samples
        )

case class PlaneColorSpec(color1: menger.common.Color, color2: Option[menger.common.Color]):
  def isSolid: Boolean = color2.isEmpty
  def isCheckered: Boolean = color2.isDefined

case class PlaneConfig(
  spec: PlaneSpec,
  colorSpec: Option[PlaneColorSpec],
  material: Option[menger.optix.Material] = None
)
