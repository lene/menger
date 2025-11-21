package menger

import com.badlogic.gdx.graphics.{Color => GdxColor}
import menger.common.{Color => CommonColor}

/** Extension methods for converting between LibGDX Color and menger.common.Color */
object ColorConversions:

  extension (gdxColor: GdxColor)
    /** Convert LibGDX Color to menger.common.Color */
    def toCommonColor: CommonColor =
      CommonColor(gdxColor.r, gdxColor.g, gdxColor.b, gdxColor.a)

  extension (commonColor: CommonColor)
    /** Convert menger.common.Color to LibGDX Color */
    def toGdxColor: GdxColor =
      new GdxColor(commonColor.r, commonColor.g, commonColor.b, commonColor.a)
