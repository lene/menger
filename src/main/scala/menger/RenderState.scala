package menger

import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.Texture

case class RenderState(
  texture: Option[Texture],
  pixmap: Option[Pixmap],
  width: Int,
  height: Int
):
  def dispose(): Unit =
    texture.foreach(_.dispose())
    pixmap.foreach(_.dispose())

  def ensurePixmap(): (RenderState, Pixmap) =
    pixmap match
      case Some(pm) => (this, pm)
      case None =>
        val newPixmap = new Pixmap(width, height, Pixmap.Format.RGBA8888)
        (copy(pixmap = Some(newPixmap)), newPixmap)

  def ensureTexture(pixmap: Pixmap): (RenderState, Texture) =
    texture match
      case Some(tex) =>
        tex.draw(pixmap, 0, 0)
        (this, tex)
      case None =>
        val newTexture = new Texture(pixmap)
        (copy(texture = Some(newTexture)), newTexture)

  def updatePixmap(rgbaBytes: Array[Byte], pixmap: Pixmap): Unit =
    pixmap.getPixels.clear()
    pixmap.getPixels.put(rgbaBytes)
    pixmap.getPixels.rewind()
