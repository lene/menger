package menger.input

import com.badlogic.gdx.InputMultiplexer
import com.badlogic.gdx.graphics.PerspectiveCamera

/**
 * Input multiplexer for LibGDX 3D rendering mode.
 *
 * Uses new abstracted handlers with LibGDXInputAdapter for type-safe
 * input event handling.
 */
class MengerInputMultiplexer(
  camera: PerspectiveCamera, eventDispatcher: EventDispatcher
) extends InputMultiplexer:

  private val gdxKeyHandler = GdxKeyHandler(camera, eventDispatcher)
  private val gdxCameraHandler = GdxCameraHandler(camera, eventDispatcher)

  private val adapter = LibGDXInputAdapter(Seq(gdxCameraHandler, gdxKeyHandler))

  addProcessor(adapter)

  /**
   * Update method called from render loop to handle continuous key presses.
   * @param deltaTime Time since last frame in seconds
   */
  def update(deltaTime: Float): Unit =
    gdxKeyHandler.update(deltaTime)
