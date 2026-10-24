package menger.input

import menger.common.InputEvent

/** Reports every input event to `onActivity` and handles none of them, so it can sit first in
  * the handler list: the preview window's idle timer sees the input without taking it from the
  * handlers behind it (menger#75). */
class ActivityHandler(onActivity: () => Unit) extends InputHandler:
  override def handleInput(event: InputEvent): Boolean =
    onActivity()
    false
