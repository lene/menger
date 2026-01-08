package menger.input

import com.badlogic.gdx.Gdx

class OptiXKeyController extends BaseKeyController:

  // Quit handling inherited from BaseKeyController (Ctrl+Q, ESC)
  // Arrow keys tracked but not used yet (future 4D rotation)

  override protected def handleEscape(): Boolean =
    Gdx.app.exit()
    true

  override protected def onRotationUpdate(): Unit = ()
