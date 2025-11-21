package menger.common

/** Simple color representation with RGBA components (0.0-1.0 range) */
case class Color(r: Float, g: Float, b: Float, a: Float = 1.0f):
  require(r >= 0 && r <= 1, s"Red component must be in [0, 1], got $r")
  require(g >= 0 && g <= 1, s"Green component must be in [0, 1], got $g")
  require(b >= 0 && b <= 1, s"Blue component must be in [0, 1], got $b")
  require(a >= 0 && a <= 1, s"Alpha component must be in [0, 1], got $a")

  def toRGBArray: Array[Float] = Array(r, g, b)
  def toRGBAArray: Array[Float] = Array(r, g, b, a)

object Color:
  val LIGHT_GRAY: Color = Color(200/255f, 200/255f, 200/255f)

  /** Create color from 0-255 integer RGB values */
  def fromRGB(r: Int, g: Int, b: Int): Color =
    Color(r / 255f, g / 255f, b / 255f)

  /** Create color from 0-255 integer RGBA values */
  def fromRGBA(r: Int, g: Int, b: Int, a: Int): Color =
    Color(r / 255f, g / 255f, b / 255f, a / 255f)

  /** Parse hex color string (formats: #RRGGBB, RRGGBB, #RRGGBBAA, or RRGGBBAA) */
  def fromHex(hex: String): Color =
    val clean = hex.stripPrefix("#")
    require(clean.length == 6 || clean.length == 8, s"Hex color must be 6 or 8 characters, got ${clean.length}")
    val r = Integer.parseInt(clean.substring(0, 2), 16)
    val g = Integer.parseInt(clean.substring(2, 4), 16)
    val b = Integer.parseInt(clean.substring(4, 6), 16)
    val a = if clean.length == 8 then Integer.parseInt(clean.substring(6, 8), 16) else 255
    fromRGBA(r, g, b, a)
