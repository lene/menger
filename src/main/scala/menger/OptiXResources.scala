package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.Texture
import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.typesafe.scalalogging.LazyLogging
import menger.optix.OptiXRenderer

@SuppressWarnings(Array("org.wartremover.warts.Var"))
class OptiXResources(renderer: OptiXRenderer, sphereRadius: Float) extends LazyLogging:

  private var batch: Option[SpriteBatch] = None
  private var texture: Option[Texture] = None
  private var pixmap: Option[Pixmap] = None
  private var currentWidth: Int = 0
  private var currentHeight: Int = 0

  def initialize(): Unit =
    logger.info("Initializing OptiX rendering resources")
    batch = Some(new SpriteBatch())

    logger.info(s"Configuring OptiX renderer with sphere radius=$sphereRadius")
    renderer.setSphere(0f, 0f, 0f, sphereRadius)
    logger.debug(s"Configured sphere: center=(0,0,0), radius=$sphereRadius")

    val eye = Array(0f, 0f, 3f)
    val lookAt = Array(0f, 0f, 0f)
    val up = Array(0f, 1f, 0f)
    val fov = 45f
    renderer.setCamera(eye, lookAt, up, fov)
    logger.debug(s"Configured camera: eye=${eye.mkString(",")}, lookAt=${lookAt.mkString(",")}, fov=$fov")

    val lightDirection = Array(-1f, -1f, -1f)
    val lightIntensity = 1.0f
    renderer.setLight(lightDirection, lightIntensity)
    logger.debug(s"Configured light: direction=${lightDirection.mkString(",")}, intensity=$lightIntensity")

  def render(rgbaBytes: Array[Byte], width: Int, height: Int): Unit =
    require(rgbaBytes.length == width * height * 4,
      s"Invalid RGBA byte array size: expected ${width * height * 4}, got ${rgbaBytes.length}")

    // Recreate texture if dimensions changed
    if width != currentWidth || height != currentHeight then
      disposeTextureAndPixmap()
      currentWidth = width
      currentHeight = height

    // Create or update pixmap
    val pm = pixmap.getOrElse {
      val newPixmap = new Pixmap(width, height, Pixmap.Format.RGBA8888)
      pixmap = Some(newPixmap)
      newPixmap
    }

    // Copy RGBA bytes to pixmap buffer
    pm.getPixels.clear()
    pm.getPixels.put(rgbaBytes)
    pm.getPixels.rewind()

    // Create or update texture
    val tex = texture match
      case Some(existingTexture) =>
        existingTexture.draw(pm, 0, 0)
        existingTexture
      case None =>
        val newTexture = new Texture(pm)
        texture = Some(newTexture)
        newTexture

    // Render texture fullscreen
    batch.foreach { b =>
      b.begin()
      // Draw texture fullscreen: from (0,0) to (screen_width, screen_height)
      // Note: LibGDX uses bottom-left origin, texture coordinates: (0,0) to (1,1)
      b.draw(tex, 0, 0, Gdx.graphics.getWidth.toFloat, Gdx.graphics.getHeight.toFloat)
      b.end()
    }

  def resize(width: Int, height: Int): Unit =
    logger.debug(s"Window resized to ${width}x${height}")
    // Note: Texture will be recreated on next updateAndRender() call
    // when OptiX renders at new dimensions

  private def disposeTextureAndPixmap(): Unit =
    texture.foreach(_.dispose())
    texture = None
    pixmap.foreach(_.dispose())
    pixmap = None

  def dispose(): Unit =
    logger.info("Disposing OptiX rendering resources")
    disposeTextureAndPixmap()
    batch.foreach(_.dispose())
    batch = None
