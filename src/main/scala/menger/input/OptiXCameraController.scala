package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.Buttons
import com.badlogic.gdx.InputAdapter
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.OptiXRenderResources
import menger.optix.CameraState
import menger.optix.OptiXRendererWrapper

class OptiXCameraController(
  rendererWrapper: OptiXRendererWrapper,
  cameraState: CameraState,
  renderResources: OptiXRenderResources,
  initialEye: Vector3,
  initialLookAt: Vector3,
  initialUp: Vector3,
  config: OrbitConfig = OrbitConfig()
) extends InputAdapter with SphericalOrbit with LazyLogging:

  // SphericalOrbit config
  override protected def orbitConfig: OrbitConfig = config

  // Camera state - stored in both Cartesian and spherical coordinates
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var eye: Vector3 = initialEye.cpy()

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var lookAt: Vector3 = initialLookAt.cpy()

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var up: Vector3 = initialUp.cpy()

  // Spherical coordinates for orbit control - implementing SphericalOrbit requirements
  private val (initAzimuth, initElevation, initDistance) = initSpherical(initialEye, initialLookAt)

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var _azimuth: Float = initAzimuth
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var _elevation: Float = initElevation
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var _distance: Float = initDistance

  override protected def azimuth: Float = _azimuth
  override protected def azimuth_=(value: Float): Unit = _azimuth = value
  override protected def elevation: Float = _elevation
  override protected def elevation_=(value: Float): Unit = _elevation = value
  override protected def distance: Float = _distance
  override protected def distance_=(value: Float): Unit = _distance = value

  // Mouse tracking state
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var lastX: Int = 0

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var lastY: Int = 0

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var isDragging: Boolean = false

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var dragButton: Int = -1

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

  private def handleOrbit(deltaX: Int, deltaY: Int): Unit =
    updateOrbit(deltaX, deltaY)
    updateEyeFromSpherical()
    updateCamera()

  private def handlePan(deltaX: Int, deltaY: Int): Unit =
    val forward = lookAt.cpy().sub(eye).nor()
    val panOffset = computePanOffset(deltaX, deltaY, forward, up)
    eye.add(panOffset)
    lookAt.add(panOffset)
    updateCamera()

  private def handleZoom(scrollAmount: Float): Unit =
    updateZoom(scrollAmount)
    updateEyeFromSpherical()
    updateCamera()

  private def updateEyeFromSpherical(): Unit =
    val newEye = sphericalToCartesian(lookAt)
    eye.set(newEye.x, newEye.y, newEye.z)

  private def updateCamera(): Unit =
    cameraState.updateCamera(rendererWrapper.renderer, eye, lookAt, up)
    renderResources.markNeedsRender()
    Gdx.graphics.requestRendering()
    logger.debug(s"Camera updated: eye=$eye, lookAt=$lookAt, distance=$distance, azimuth=$azimuth, elevation=$elevation")
