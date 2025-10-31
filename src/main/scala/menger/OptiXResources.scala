package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.Texture
import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.typesafe.scalalogging.LazyLogging

/**
 * Manages resources for displaying OptiX-rendered 2D images in LibGDX window.
 *
 * OptiX renders to RGBA byte array, which is converted to a Pixmap, then to a Texture,
 * and finally displayed fullscreen using SpriteBatch.
 *
 * This class handles the 2D rendering pipeline:
 * 1. Convert RGBA bytes to Pixmap
 * 2. Update/create Texture from Pixmap
 * 3. Render texture fullscreen via SpriteBatch
 * 4. Handle window resize (recreate texture at new size)
 * 5. Cleanup on disposal
 */
// TODO: Replace var with immutable state management (e.g., State monad, Ref, or AtomicReference)
// Current mutable state is for LibGDX resource lifecycle management
@SuppressWarnings(Array("org.wartremover.warts.Var"))
class OptiXResources extends LazyLogging:

  private var batch: Option[SpriteBatch] = None
  private var texture: Option[Texture] = None
  private var pixmap: Option[Pixmap] = None
  private var currentWidth: Int = 0
  private var currentHeight: Int = 0

  /**
   * Initialize rendering resources (SpriteBatch).
   * Called once during engine creation.
   */
  def initialize(): Unit =
    logger.info("Initializing OptiX rendering resources")
    batch = Some(new SpriteBatch())

  /**
   * Update texture from RGBA byte array and render fullscreen.
   *
   * @param rgbaBytes RGBA byte array from OptiX renderer (4 bytes per pixel)
   * @param width Image width in pixels
   * @param height Image height in pixels
   */
  def updateAndRender(rgbaBytes: Array[Byte], width: Int, height: Int): Unit =
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

  /**
   * Handle window resize.
   * Called by LibGDX when window size changes.
   *
   * @param width New window width
   * @param height New window height
   */
  def resize(width: Int, height: Int): Unit =
    logger.debug(s"Window resized to ${width}x${height}")
    // Note: Texture will be recreated on next updateAndRender() call
    // when OptiX renders at new dimensions

  /**
   * Dispose of texture and pixmap resources.
   */
  private def disposeTextureAndPixmap(): Unit =
    texture.foreach(_.dispose())
    texture = None
    pixmap.foreach(_.dispose())
    pixmap = None

  /**
   * Dispose all resources.
   * Called when engine shuts down.
   */
  def dispose(): Unit =
    logger.info("Disposing OptiX rendering resources")
    disposeTextureAndPixmap()
    batch.foreach(_.dispose())
    batch = None
