package menger

import com.badlogic.gdx.graphics.{Color => GdxColor}
import menger.common.Const
import menger.common.{Color => CommonColor}

object ColorConversions:

  extension (gdxColor: GdxColor)
    def toCommonColor: CommonColor =
      CommonColor(gdxColor.r, gdxColor.g, gdxColor.b, gdxColor.a)

  extension (commonColor: CommonColor)
    def toGdxColor: GdxColor =
      new GdxColor(commonColor.r, commonColor.g, commonColor.b, commonColor.a)

  def rgbIntsToColor(parts: Array[Int]): GdxColor =
    val Array(r, g, b, a) = parts.map(_ / Const.rgbMaxValueFloat).padTo(4, 1f)
    GdxColor(r, g, b, a)
