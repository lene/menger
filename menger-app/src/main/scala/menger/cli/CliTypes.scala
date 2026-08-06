package menger.cli

import menger.common.Color
import menger.common.Vector

// Domain types moved to menger.common — re-exported here for backward compatibility
// within the cli package (converters, CliValidation, etc. use unqualified names).
export menger.common.Axis
export menger.common.PlaneSpec
export menger.common.PlaneColorSpec
export menger.common.FogSpec

enum LightType:
  case DIRECTIONAL, POINT, AREA

enum AreaLightShape:
  case DISK

case class LightSpec(
  lightType: LightType,
  position: Vector[3],
  intensity: Float,
  color: Color,
  normal: Vector[3] = Vector[3](0f, -1f, 0f),
  radius: Float = 1.0f,
  shape: AreaLightShape = AreaLightShape.DISK,
  shadowSamples: Int = 4
)

object LightSpec:
  /** Convert CLI LightSpec to menger.common.Light. */
  def toCommonLight(spec: LightSpec): menger.common.Light =
    val pos   = spec.position
    val clr   = spec.color
    spec.lightType match
      case LightType.DIRECTIONAL =>
        menger.common.Light.Directional(pos, clr, spec.intensity)
      case LightType.POINT =>
        menger.common.Light.Point(pos, clr, spec.intensity)
      case LightType.AREA =>
        menger.common.Light.Area(pos, spec.normal, spec.radius,
          menger.common.AreaLightShape.Disk, clr, spec.intensity, spec.shadowSamples)

  /** Convert common.Light to CLI LightSpec (e.g. for round-trip tests). */
  def fromCommonLight(light: menger.common.Light): LightSpec =
    light match
      case menger.common.Light.Directional(direction, clr, intensity) =>
        LightSpec(LightType.DIRECTIONAL, direction, intensity, clr)
      case menger.common.Light.Point(position, clr, intensity) =>
        LightSpec(LightType.POINT, position, intensity, clr)
      case menger.common.Light.Area(position, normal, radius, shape, clr, intensity, samples) =>
        val cliShape = shape match
          case menger.common.AreaLightShape.Disk => AreaLightShape.DISK
        LightSpec(
          LightType.AREA,
          position,
          intensity,
          clr,
          normal = normal,
          radius = radius,
          shape = cliShape,
          shadowSamples = samples
        )


// PlaneConfig moved to menger.config — re-exported here so that cli-internal code
// that references PlaneConfig by unqualified name continues to compile.
export menger.config.PlaneConfig

