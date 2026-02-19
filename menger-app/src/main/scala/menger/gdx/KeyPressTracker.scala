package menger.gdx

import menger.common.Key

/**
 * Tracks which keys are currently held down.
 *
 * Encapsulates the mutable Map[Key, Boolean] pattern used in both
 * OptiXKeyHandler and GdxKeyHandler, eliminating duplicate var declarations.
 */
class KeyPressTracker:
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var pressed: Map[Key, Boolean] = Map().withDefaultValue(false)

  def press(key: Key): Unit        = pressed = pressed.updated(key, true)
  def release(key: Key): Unit      = pressed = pressed.updated(key, false)
  def isPressed(key: Key): Boolean = pressed(key)
  def anyPressed: Boolean          = pressed.values.exists(identity)
