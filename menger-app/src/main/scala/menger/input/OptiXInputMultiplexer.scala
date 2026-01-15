package menger.input

import com.badlogic.gdx.InputMultiplexer

/**
 * Input multiplexer for OptiX ray-traced rendering mode.
 *
 * Uses new abstracted handlers with LibGDXInputAdapter for type-safe
 * input event handling.
 */
class OptiXInputMultiplexer(
  cameraHandler: OptiXCameraHandler,
  keyHandler: OptiXKeyHandler
) extends InputMultiplexer:

  private val adapter = LibGDXInputAdapter(Seq(cameraHandler, keyHandler))

  addProcessor(adapter)

  /**
   * Update method called from render loop to handle continuous key presses.
   * @param deltaTime Time since last frame in seconds
   */
  def update(deltaTime: Float): Unit =
    keyHandler.update(deltaTime)
