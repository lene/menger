package menger.input

import menger.common.Key
import menger.common.ModifierState

class PreviewKeyHandler(
  onStep: Float => Unit,
  onTogglePlay: () => Unit,
  onJumpStart: () => Unit,
  onJumpEnd: () => Unit
) extends KeyHandler:

  override protected def handleKeyPress(key: Key, modifiers: ModifierState): Boolean =
    key match
      case Key.Left  if !modifiers.shift => onStep(-0.01f); true
      case Key.Right if !modifiers.shift => onStep( 0.01f); true
      case Key.Left  if  modifiers.shift => onStep(-0.1f);  true
      case Key.Right if  modifiers.shift => onStep( 0.1f);  true
      case Key.Home  => onJumpStart(); true
      case Key.End   => onJumpEnd();   true
      case Key.Space => onTogglePlay(); true
      case Key.Q if modifiers.ctrl => GdxRuntime.exit(); true
      case _ => false

  override protected def handleKeyRelease(key: Key, modifiers: ModifierState): Boolean = false
