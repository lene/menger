package menger.input

import menger.common.Key
import menger.gdx.KeyPressTracker

/** Shared key-rotation angle calculation used by GdxKeyHandler and OptiXKeyHandler. */
trait KeyRotation:
  protected def rotatePressed: KeyPressTracker
  protected def rotateAngle: Float

  protected val factor: Map[Key, Int] = Map(
    Key.Right   -> -1, Key.Left  -> 1,
    Key.Up      ->  1, Key.Down  -> -1,
    Key.PageUp  ->  1, Key.PageDown -> -1
  )

  protected def angle(delta: Float, keys: Seq[Key]): Float =
    delta * rotateAngle * keys.find(rotatePressed.isPressed).map(factor(_)).getOrElse(0)
