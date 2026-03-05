package menger.input

import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.math.Vector3
import menger.RotationProjectionParameters
import menger.common.Const
import menger.common.Key
import menger.common.ModifierState
import menger.gdx.GdxRuntime
import menger.gdx.KeyPressTracker

/**
 * Keyboard input handler for LibGDX 3D rendering mode.
 *
 * Handles:
 * - Arrow keys without modifiers: rotate camera around origin
 * - Arrow keys with Shift: dispatch 4D rotation events
 * - Escape: reset camera to default position
 * - Ctrl+Q: quit application
 */
class GdxKeyHandler(
  camera: PerspectiveCamera,
  dispatcher: EventDispatcher
) extends KeyHandler with KeyRotation:

  private val defaultPos       = camera.position.cpy
  private val defaultDirection = camera.direction.cpy
  private val defaultUp        = camera.up.cpy
  protected val rotateAngle      = Const.Input.defaultRotateAngle
  protected val rotatePressed    = KeyPressTracker()

  override protected def handleKeyPress(key: Key, modifiers: ModifierState): Boolean =
    key match
      case Key.Left | Key.Right | Key.Up | Key.Down | Key.PageUp | Key.PageDown =>
        rotatePressed.press(key)
        false
      case Key.Escape =>
        resetCamera()
        false
      case Key.Q if modifiers.ctrl =>
        GdxRuntime.exit()
        true
      case _ => false

  override protected def handleKeyRelease(key: Key, modifiers: ModifierState): Boolean =
    key match
      case Key.Left | Key.Right | Key.Up | Key.Down | Key.PageUp | Key.PageDown =>
        rotatePressed.release(key)
        false
      case _ => false

  /**
   * Called each frame to handle continuous rotation while keys are held.
   * Must be called from render loop.
   */
  def update(deltaTime: Float): Unit =
    if rotatePressed.anyPressed then
      if !(modifierState.shift || modifierState.ctrl || modifierState.alt) then
        onNoModifiersPressed(deltaTime)
      else if modifierState.shift then
        onShiftPressed(deltaTime)

  private val origin = Vector3.Zero

  private def onNoModifiersPressed(delta: Float): Unit =
    // Algorithm from LibGDX CameraInputController
    // https://github.com/libgdx/libgdx/blob/master/gdx/src/com/badlogic/gdx/graphics/g3d/utils/CameraInputController.java#L187
    camera.rotateAround(origin, Vector3.Y, angle(delta, Seq(Key.Right, Key.Left)))
    val tmpXZ = Vector3()
    tmpXZ.set(camera.direction).crs(camera.up).y = 0f
    camera.rotateAround(origin, tmpXZ.nor, angle(delta, Seq(Key.Up, Key.Down)))
    camera.update()

  private def onShiftPressed(delta: Float): Unit =
    dispatcher.notifyObservers(
      RotationProjectionParameters(
        angle(delta, Seq(Key.Left, Key.Right)),
        angle(delta, Seq(Key.Up, Key.Down)),
        angle(delta, Seq(Key.PageUp, Key.PageDown))
      )
    )

  private def resetCamera(): Unit =
    camera.position.set(defaultPos)
    camera.direction.set(defaultDirection)
    camera.up.set(defaultUp)
    camera.lookAt(0, 0, 0)
    camera.update()
