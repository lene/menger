package menger.cli

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3

enum Axis:
  case X, Y, Z

case class PlaneSpec(axis: Axis, positive: Boolean, value: Float)

enum LightType:
  case DIRECTIONAL, POINT

case class LightSpec(lightType: LightType, position: Vector3, intensity: Float, color: Color)

case class PlaneColorSpec(color1: menger.common.Color, color2: Option[menger.common.Color]):
  def isSolid: Boolean = color2.isEmpty
  def isCheckered: Boolean = color2.isDefined
