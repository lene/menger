package menger.input

import scala.math._

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.Buttons
import com.badlogic.gdx.InputAdapter
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.OptiXRenderResources
import menger.OptiXResources
import menger.common.Const


case class CameraControlConfig(
  orbitSensitivity: Float = 0.3f,
  panSensitivity: Float = 0.005f,
  zoomSensitivity: Float = 0.1f,
  minDistance: Float = 0.5f,
  maxDistance: Float = 20.0f,
  minElevation: Float = -89.0f,
  maxElevation: Float = 89.0f
)

object CameraControlConfig:
  
  def default: CameraControlConfig = CameraControlConfig()


class OptiXCameraController(
  optiXResources: OptiXResources,
  renderResources: OptiXRenderResources,
  initialEye: Vector3,
  initialLookAt: Vector3,
  initialUp: Vector3,
  config: CameraControlConfig = CameraControlConfig.default
) extends InputAdapter with LazyLogging:

  // Camera state - stored in both Cartesian and spherical coordinates
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var eye: Vector3 = initialEye.cpy()

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var lookAt: Vector3 = initialLookAt.cpy()

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var up: Vector3 = initialUp.cpy()

  // Spherical coordinates for orbit control
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var distance: Float = eye.cpy().sub(lookAt).len()

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var azimuth: Float = computeInitialAzimuth()

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var elevation: Float = computeInitialElevation()

  // Mouse tracking state
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var lastX: Int = 0

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var lastY: Int = 0

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var isDragging: Boolean = false

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var dragButton: Int = -1

  // Compute initial azimuth angle from eye and lookAt positions
  private def computeInitialAzimuth(): Float =
    val dir = eye.cpy().sub(lookAt)
    val azimuthRad = atan2(dir.x.toDouble, dir.z.toDouble).toFloat
    Const.radiansToDegrees(azimuthRad)

  // Compute initial elevation angle from eye and lookAt positions
  private def computeInitialElevation(): Float =
    val dir = eye.cpy().sub(lookAt)
    val horizontalDist = sqrt(dir.x * dir.x + dir.z * dir.z).toFloat
    val elevationRad = atan2(dir.y.toDouble, horizontalDist.toDouble).toFloat
    Const.radiansToDegrees(elevationRad)

  override def touchDown(screenX: Int, screenY: Int, pointer: Int, button: Int): Boolean =
    lastX = screenX
    lastY = screenY
    isDragging = true
    dragButton = button
    true

  override def touchUp(screenX: Int, screenY: Int, pointer: Int, button: Int): Boolean =
    isDragging = false
    dragButton = -1
    true

  override def touchDragged(screenX: Int, screenY: Int, pointer: Int): Boolean =
    if !isDragging then
      false
    else
      val deltaX = screenX - lastX
      val deltaY = screenY - lastY
      lastX = screenX
      lastY = screenY

      dragButton match
        case Buttons.LEFT => handleOrbit(deltaX, deltaY)
        case Buttons.RIGHT => handlePan(deltaX, deltaY)
        case _ => // Ignore other buttons

      true

  override def scrolled(amountX: Float, amountY: Float): Boolean =
    handleZoom(amountY)
    true

  // Orbit camera: rotate around lookAt point based on mouse delta
  private def handleOrbit(deltaX: Int, deltaY: Int): Unit =
    // Update spherical coordinates
    azimuth += deltaX * config.orbitSensitivity
    elevation -= deltaY * config.orbitSensitivity  // Invert Y for natural feel

    // Clamp elevation to prevent gimbal lock
    elevation = elevation.max(config.minElevation).min(config.maxElevation)

    // Convert spherical to Cartesian and update eye position
    updateEyeFromSpherical()
    updateCamera()

  // Pan camera: translate both eye and lookAt in screen space
  private def handlePan(deltaX: Int, deltaY: Int): Unit =
    // Compute camera right and up vectors
    val forward = lookAt.cpy().sub(eye).nor()
    val right = forward.cpy().crs(up).nor()
    val camUp = right.cpy().crs(forward).nor()

    // Scale pan delta by distance for natural feel
    val panScale = distance * config.panSensitivity

    // Translate both eye and lookAt
    val deltaRight = right.scl(deltaX * panScale)
    val deltaUp = camUp.scl(-deltaY * panScale) // Invert Y for natural feel

    eye.add(deltaRight).add(deltaUp)
    lookAt.add(deltaRight).add(deltaUp)

    updateCamera()

  // Zoom camera: move eye closer/farther from lookAt along view direction
  private def handleZoom(scrollAmount: Float): Unit =
    // Update distance with exponential scaling for smooth zoom
    val zoomFactor = 1.0f + (scrollAmount * config.zoomSensitivity)
    distance *= zoomFactor

    // Clamp distance to prevent camera entering geometry or going too far
    distance = distance.max(config.minDistance).min(config.maxDistance)

    // Update eye position from new distance
    updateEyeFromSpherical()
    updateCamera()

  // Convert spherical coordinates (azimuth, elevation, distance) to Cartesian eye position
  private def updateEyeFromSpherical(): Unit =
    val azimuthRad = Const.degreesToRadians(azimuth)
    val elevationRad = Const.degreesToRadians(elevation)

    // Spherical to Cartesian conversion
    val cosElev = cos(elevationRad.toDouble).toFloat
    val x = lookAt.x + distance * sin(azimuthRad.toDouble).toFloat * cosElev
    val y = lookAt.y + distance * sin(elevationRad.toDouble).toFloat
    val z = lookAt.z + distance * cos(azimuthRad.toDouble).toFloat * cosElev

    eye.set(x, y, z)

  // Update OptiX camera and trigger re-render
  private def updateCamera(): Unit =
    optiXResources.updateCamera(eye, lookAt, up)
    renderResources.markNeedsRender()
    Gdx.graphics.requestRendering()
    logger.debug(s"Camera updated: eye=$eye, lookAt=$lookAt, distance=$distance, azimuth=$azimuth, elevation=$elevation")
