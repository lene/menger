package menger

import java.util.concurrent.atomic.AtomicReference

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.Axis
import menger.LightSpec
import menger.LightType
import menger.PlaneSpec
import menger.common.ImageSize
import menger.common.{Light => CommonLight}
import menger.common.{Vector => CommonVector}
import menger.optix.OptiXRenderer

class OptiXResources(
  configureGeometry: Try[OptiXRenderer => Unit],
  cameraPos: Vector3,
  cameraLookat: Vector3,
  cameraUp: Vector3,
  planeSpec: PlaneSpec,
  lights: Option[List[LightSpec]] = None
) extends LazyLogging:

  private val _rendererRef = new AtomicReference[Option[OptiXRenderer]](None)

  private def renderer: OptiXRenderer =
    _rendererRef.get() match
      case Some(r) => r
      case None =>
        val r = initializeRenderer
        _rendererRef.set(Some(r))
        r

  private def errorExit(message: String): Unit =
    logger.error(message)
    System.exit(1)

  private def initializeRenderer: OptiXRenderer =
    OptiXRenderer().ensureAvailable().recover:
      case exception =>
        errorExit(exception.getMessage)
        // errorExit calls System.exit(1), so this is unreachable
        // Return dummy value to satisfy type checker (never executed)
        OptiXRenderer()
    .get  // Safe because recover always returns Success

  def initialize(): Unit =
    logger.debug("Configuring OptiX scene")
    configureGeometry match
      case Success(config) => config(renderer)
      case Failure(exception) => errorExit(
        s"Invalid geometry configuration: ${exception.getMessage}"
      )
    createCamera()
    createLights()
    configurePlane()

  private def createCamera(): Unit =
    val eye = CommonVector[3](cameraPos.x, cameraPos.y, cameraPos.z)
    val lookAt = CommonVector[3](cameraLookat.x, cameraLookat.y, cameraLookat.z)
    val up = CommonVector[3](cameraUp.x, cameraUp.y, cameraUp.z)
    val horizontalFov = 45f
    renderer.setCamera(eye, lookAt, up, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Configured camera: eye=(${eye(0)},${eye(1)},${eye(2)}), lookAt=(${lookAt(0)},${lookAt(1)},${lookAt(2)}), up=(${up(0)},${up(1)},${up(2)}), horizontalFOV=$horizontalFov")

  private def createLights(): Unit =
    lights match
      case Some(lightSpecs) =>
        val lightSeq = lightSpecs.map(convertLightSpec)
        Try(renderer.setLights(lightSeq)) match
          case Success(_) =>
            logger.debug(s"Configured ${lightSeq.length} light(s) from CLI specification")
          case Failure(exception) =>
            errorExit(s"Failed to configure lights: ${exception.getMessage}")
      case None =>
        // Default single directional light (backward compatibility)
        val lightDirection = CommonVector[3](-1f, 1f, -1f)  // Light from upper-left-back (Y positive = from above)
        val lightIntensity = 1.0f
        renderer.setLight(lightDirection, lightIntensity)
        logger.debug(s"Configured default light: direction=(${lightDirection(0)},${lightDirection(1)},${lightDirection(2)}), intensity=$lightIntensity")

  private def convertLightSpec(spec: LightSpec): CommonLight =
    val position = menger.common.Vector[3](spec.position.x, spec.position.y, spec.position.z)
    val color = menger.common.Vector[3](spec.color.r, spec.color.g, spec.color.b)

    spec.lightType match
      case LightType.DIRECTIONAL => CommonLight.Directional(position, color, spec.intensity)
      case LightType.POINT => CommonLight.Point(position, color, spec.intensity)

  private def configurePlane(): Unit =
    val axisInt = planeSpec.axis match
      case Axis.X => 0
      case Axis.Y => 1
      case Axis.Z => 2
    renderer.setPlane(axisInt, planeSpec.positive, planeSpec.value)
    val axisName = planeSpec.axis.toString.toLowerCase
    val sign = if planeSpec.positive then "+" else "-"
    logger.debug(s"Configured plane: ${sign}${axisName}:${planeSpec.value}")

  def setSphereColor(r: Float, g: Float, b: Float, a: Float = 1.0f): Unit =
    renderer.setSphereColor(r, g, b, a)
    logger.debug(s"Configured sphere color: RGBA=($r, $g, $b, $a)")

  def setIOR(ior: Float): Unit =
    renderer.setIOR(ior)
    logger.debug(s"Configured index of refraction: IOR=$ior")

  def setScale(scale: Float): Unit =
    renderer.setScale(scale)
    logger.debug(s"Configured scale parameter: scale=$scale")

  def setShadows(enabled: Boolean): Unit =
    renderer.setShadows(enabled)
    logger.debug(s"Configured shadow rays: enabled=$enabled")

  def renderScene(size: ImageSize): Array[Byte] =
    logger.debug(s"[OptiXResources] renderScene: rendering at ${size.width}x${size.height}")
    renderer.render(size).getOrElse:
      logger.error("OptiX rendering failed - returned None")
      Array.emptyByteArray

  def renderSceneWithStats(size: ImageSize): menger.optix.RenderResult =
    renderer.renderWithStats(size)

  def updateCamera(eye: Vector3, lookAt: Vector3, up: Vector3): Unit =
    val eyeVec = CommonVector[3](eye.x, eye.y, eye.z)
    val lookAtVec = CommonVector[3](lookAt.x, lookAt.y, lookAt.z)
    val upVec = CommonVector[3](up.x, up.y, up.z)
    val horizontalFov = 45f
    renderer.setCamera(eyeVec, lookAtVec, upVec, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Updated camera: eye=(${eyeVec(0)},${eyeVec(1)},${eyeVec(2)}), lookAt=(${lookAtVec(0)},${lookAtVec(1)},${lookAtVec(2)}), up=(${upVec(0)},${upVec(1)},${upVec(2)})")

  def updateCameraAspectRatio(size: ImageSize): Unit =
    val horizontalFov = 45f  // Fixed horizontal FOV in degrees (aspect-ratio independent)

    // Update cached image dimensions BEFORE calling setCamera
    renderer.updateImageDimensions(size)

    val eye = CommonVector[3](cameraPos.x, cameraPos.y, cameraPos.z)
    val lookAt = CommonVector[3](cameraLookat.x, cameraLookat.y, cameraLookat.z)
    val up = CommonVector[3](cameraUp.x, cameraUp.y, cameraUp.z)

    renderer.setCamera(eye, lookAt, up, horizontalFovDegrees = horizontalFov)

  def dispose(): Unit =
    _rendererRef.get().foreach { r =>
      logger.debug("Disposing OptiX renderer")
      r.dispose()
    }
