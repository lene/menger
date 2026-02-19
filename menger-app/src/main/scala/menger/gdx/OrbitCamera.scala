package menger.gdx

import com.badlogic.gdx.math.Vector3
import menger.common.MouseButton
import menger.common.ScreenCoords
import menger.input.OrbitConfig
import menger.input.SphericalCoords
import menger.input.SphericalOrbit

/** Consolidated drag state — reduces multiple vars to a single Option */
private case class CameraDragState(lastPos: ScreenCoords, button: MouseButton)

/**
 * Encapsulates all mutable camera state for a spherical orbit camera.
 *
 * Owns 5 vars (eye, lookAt, up, spherical, dragState) that were previously
 * spread across OptiXCameraHandler. Provides orbit, pan, zoom, and drag
 * management while hiding mutation behind a clean API.
 */
class OrbitCamera(
  initialEye: Vector3,
  initialLookAt: Vector3,
  initialUp: Vector3,
  config: OrbitConfig = OrbitConfig()
) extends SphericalOrbit:

  override protected def orbitConfig: OrbitConfig = config

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var eye: Vector3 = initialEye.cpy()

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var lookAt: Vector3 = initialLookAt.cpy()

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var up: Vector3 = initialUp.cpy()

  private val (initAzimuth, initElevation, initDistance) = initSpherical(initialEye, initialLookAt)

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var spherical: SphericalCoords = SphericalCoords(initAzimuth, initElevation, initDistance)

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var dragState: Option[CameraDragState] = None

  // SphericalOrbit abstract members
  override protected def azimuth: Float             = spherical.azimuth
  override protected def azimuth_=(v: Float): Unit  = spherical = spherical.copy(azimuth = v)
  override protected def elevation: Float           = spherical.elevation
  override protected def elevation_=(v: Float): Unit = spherical = spherical.copy(elevation = v)
  override protected def distance: Float            = spherical.distance
  override protected def distance_=(v: Float): Unit = spherical = spherical.copy(distance = v)

  // Read-only accessors — return copies to prevent external mutation
  def currentEye: Vector3    = eye.cpy()
  def currentLookAt: Vector3 = lookAt.cpy()
  def currentUp: Vector3     = up.cpy()

  /** Apply orbit (azimuth/elevation) from mouse drag delta. */
  def orbit(deltaX: Int, deltaY: Int): Unit =
    updateOrbit(deltaX, deltaY)
    updateEyeFromSpherical()

  /** Apply pan (eye + lookAt shift) from mouse drag delta. */
  def pan(deltaX: Int, deltaY: Int): Unit =
    val forward   = lookAt.cpy().sub(eye).nor()
    val panOffset = computePanOffset(deltaX, deltaY, forward, up)
    eye.add(panOffset)
    lookAt.add(panOffset)

  /** Apply zoom (distance change) from scroll amount. */
  def zoom(scrollAmount: Float): Unit =
    updateZoom(scrollAmount)
    updateEyeFromSpherical()

  /** Begin a drag gesture. */
  def startDrag(pos: ScreenCoords, button: MouseButton): Unit =
    dragState = Some(CameraDragState(pos, button))

  /** End the current drag gesture. */
  def endDrag(): Unit = dragState = None

  /**
   * Move an ongoing drag and return the delta.
   *
   * Returns `Some((deltaX, deltaY, button))` if a drag is in progress and
   * the drag state has been updated, `None` if no drag is active.
   */
  def moveDrag(pos: ScreenCoords): Option[(Int, Int, MouseButton)] =
    dragState match
      case None => None
      case Some(state) =>
        val deltaX = pos.x - state.lastPos.x
        val deltaY = pos.y - state.lastPos.y
        dragState = Some(state.copy(lastPos = pos))
        Some((deltaX, deltaY, state.button))

  private def updateEyeFromSpherical(): Unit =
    val newEye = sphericalToCartesian(lookAt)
    eye.set(newEye.x, newEye.y, newEye.z)
