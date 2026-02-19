package menger.gdx

import menger.common.ScreenCoords

/**
 * Tracks the origin of an ongoing drag gesture.
 *
 * Encapsulates the mutable ScreenCoords var from GdxCameraHandler.
 */
class DragTracker:
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var _origin: ScreenCoords = ScreenCoords(0, 0)

  def start(pos: ScreenCoords): Unit = _origin = pos
  def origin: ScreenCoords           = _origin
