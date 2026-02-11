package menger.dsl

import menger.common.Color as CommonColor

/** Color with hex string parsing for DSL */
case class Color(r: Float, g: Float, b: Float, a: Float = 1.0f):
  require(r >= 0f && r <= 1f, s"Red component must be in [0, 1], got $r")
  require(g >= 0f && g <= 1f, s"Green component must be in [0, 1], got $g")
  require(b >= 0f && b <= 1f, s"Blue component must be in [0, 1], got $b")
  require(a >= 0f && a <= 1f, s"Alpha component must be in [0, 1], got $a")

  def toCommonColor: CommonColor = CommonColor(r, g, b, a)

object Color:
  val White = Color(1f, 1f, 1f)
  val Black = Color(0f, 0f, 0f)
  val Red = Color(1f, 0f, 0f)
  val Green = Color(0f, 1f, 0f)
  val Blue = Color(0f, 0f, 1f)
  val Yellow = Color(1f, 1f, 0f)
  val Cyan = Color(0f, 1f, 1f)
  val Magenta = Color(1f, 0f, 1f)
  val Gray = Color(0.5f, 0.5f, 0.5f)

  /** Parse hex color string (e.g., "#FF0000" or "FF0000" or "#FF0000FF") */
  def apply(hex: String): Color =
    val cleanHex = if hex.startsWith("#") then hex.substring(1) else hex
    require(
      cleanHex.length == 6 || cleanHex.length == 8,
      s"Hex color must be 6 or 8 characters, got ${cleanHex.length}: '$hex'"
    )
    require(
      cleanHex.forall(c => c.isDigit || ('a' <= c && c <= 'f') || ('A' <= c && c <= 'F')),
      s"Hex color must contain only hex digits, got: '$hex'"
    )

    val r = Integer.parseInt(cleanHex.substring(0, 2), 16) / 255f
    val g = Integer.parseInt(cleanHex.substring(2, 4), 16) / 255f
    val b = Integer.parseInt(cleanHex.substring(4, 6), 16) / 255f
    val a = if cleanHex.length == 8 then Integer.parseInt(cleanHex.substring(6, 8), 16) / 255f else 1f
    Color(r, g, b, a)

  // Implicit conversion for string literals
  given Conversion[String, Color] = Color(_)
