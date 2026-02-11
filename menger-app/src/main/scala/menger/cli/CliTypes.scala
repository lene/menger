package menger.cli

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3

enum Axis:
  case X, Y, Z

case class PlaneSpec(axis: Axis, positive: Boolean, value: Float)

enum LightType:
  case DIRECTIONAL, POINT

case class LightSpec(lightType: LightType, position: Vector3, intensity: Float, color: Color)

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

case class PlaneColorSpec(color1: menger.common.Color, color2: Option[menger.common.Color]):
  def isSolid: Boolean = color2.isEmpty
  def isCheckered: Boolean = color2.isDefined
