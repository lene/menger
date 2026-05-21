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
  /** Convert CLI LightSpec to menger.common.Light. */
  def toCommonLight(spec: LightSpec): menger.common.Light =
    val pos   = menger.common.Vector[3](spec.position.x, spec.position.y, spec.position.z)
    val clr   = menger.common.Color(spec.color.r, spec.color.g, spec.color.b, spec.color.a)
    spec.lightType match
      case LightType.DIRECTIONAL =>
        menger.common.Light.Directional(pos, clr, spec.intensity)
      case LightType.POINT =>
        menger.common.Light.Point(pos, clr, spec.intensity)
      case LightType.AREA =>
        val normal = menger.common.Vector[3](spec.normal.x, spec.normal.y, spec.normal.z)
        menger.common.Light.Area(pos, normal, spec.radius,
          menger.common.AreaLightShape.Disk, clr, spec.intensity, spec.shadowSamples)

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
        val cliShape = shape match
          case menger.common.AreaLightShape.Disk => AreaLightShape.DISK
        LightSpec(
          LightType.AREA,
          new Vector3(position(0), position(1), position(2)),
          intensity,
          new Color(clr.r, clr.g, clr.b, clr.a),
          normal = new Vector3(normal(0), normal(1), normal(2)),
          radius = radius,
          shape = cliShape,
          shadowSamples = samples
        )

case class FogSpec(density: Float, color: menger.common.Color = menger.common.Color(0.8f, 0.8f, 0.9f))

case class PlaneColorSpec(color1: menger.common.Color, color2: Option[menger.common.Color]):
  def isSolid: Boolean = color2.isEmpty
  def isCheckered: Boolean = color2.isDefined

case class PlaneConfig(
  spec: PlaneSpec,
  colorSpec: Option[PlaneColorSpec],
  material: Option[menger.optix.Material] = None
)
