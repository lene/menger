package menger

import menger.common.Color
import menger.common.Const

object ColorConversions:

  // GDX's Color constructor silently clamped out-of-range components to [0, 1];
  // menger.common.Color validates instead (require), so clamp explicitly here to
  // preserve prior behavior for callers that don't pre-validate (e.g. light color
  // parsing in EnvironmentConverters). take(4) keeps this total on 5+ component
  // input (CR-12) rather than pattern-matching a fixed-size Array.
  def rgbIntsToColor(parts: Array[Int]): Color =
    val floats =
      parts.take(4).map(_ / Const.rgbMaxValueFloat).map(v => v.max(0f).min(1f)).padTo(4, 1f)
    Color(floats(0), floats(1), floats(2), floats(3))
