package menger.input

import com.badlogic.gdx.Gdx
import com.typesafe.scalalogging.LazyLogging
import menger.RotationProjectionParameters
import menger.common.Const
import menger.common.Key
import menger.common.ModifierState

/**
 * Keyboard input handler for OptiX ray-traced rendering mode.
 *
 * Handles:
 * - Arrow keys with Shift: dispatch 4D rotation events
 * - Escape: exit application
 * - Ctrl+Q: quit application
 *
 * Note: This handler only processes Shift-modified keys for 4D rotation.
 * Unlike GdxKeyHandler, it does not handle 3D camera rotation.
 */
class OptiXKeyHandler(dispatcher: EventDispatcher) extends KeyHandler with LazyLogging:

  private val rotateAngle = Const.Input.defaultRotateAngle

  /** Track which rotation keys are currently pressed */
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rotatePressed: Map[Key, Boolean] = Map().withDefaultValue(false)

  override protected def handleKeyPress(key: Key, modifiers: ModifierState): Boolean =
    key match
      case Key.Left | Key.Right | Key.Up | Key.Down | Key.PageUp | Key.PageDown =>
        rotatePressed = rotatePressed.updated(key, true)
        false
      case Key.Escape =>
        // scalafix:off DisableSyntax.null
        if Gdx.app != null then Gdx.app.exit()
        // scalafix:on DisableSyntax.null
        true
      case Key.Q if modifiers.ctrl =>
        // scalafix:off DisableSyntax.null
        if Gdx.app != null then Gdx.app.exit()
        // scalafix:on DisableSyntax.null
        true
      case _ => false

  override protected def handleKeyRelease(key: Key, modifiers: ModifierState): Boolean =
    key match
      case Key.Left | Key.Right | Key.Up | Key.Down | Key.PageUp | Key.PageDown =>
        rotatePressed = rotatePressed.updated(key, false)
        false
      case _ => false

  /**
   * Called each frame to handle continuous rotation while keys are held.
   * Must be called from render loop.
   */
  def update(deltaTime: Float): Unit =
    if rotatePressed.values.exists(identity) && modifierState.shift then
      // Check if Gdx is initialized (for test compatibility)
      // scalafix:off DisableSyntax.null
      if Gdx.graphics != null then
        // scalafix:on DisableSyntax.null
        logger.debug(s"Shift pressed with rotation keys, delta=$deltaTime")
        onShiftPressed(deltaTime)

  private def onShiftPressed(delta: Float): Unit =
    val rotXW = angle(delta, Seq(Key.Left, Key.Right))
    val rotYW = angle(delta, Seq(Key.Up, Key.Down))
    val rotZW = angle(delta, Seq(Key.PageUp, Key.PageDown))
    logger.debug(s"Dispatching 4D rotation event: rotXW=$rotXW, rotYW=$rotYW, rotZW=$rotZW")
    dispatcher.notifyObservers(RotationProjectionParameters(rotXW, rotYW, rotZW))

  private val factor = Map(
    Key.Right -> -1, Key.Left -> 1,
    Key.Up -> 1, Key.Down -> -1,
    Key.PageUp -> 1, Key.PageDown -> -1
  )

  private def angle(delta: Float, keys: Seq[Key]): Float =
    delta * rotateAngle * keys.find(rotatePressed).map(factor(_)).getOrElse(0)
