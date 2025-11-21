package menger

import java.util.concurrent.atomic.AtomicReference

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.Axis
import menger.ColorConversions.toCommonColor
import menger.LightSpec
import menger.LightType
import menger.PlaneColorSpec
import menger.PlaneSpec
import menger.common.Color
import menger.common.ImageSize
import menger.common.Light
import menger.common.Vector
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
    val eye = Vector[3](cameraPos.x, cameraPos.y, cameraPos.z)
    val lookAt = Vector[3](cameraLookat.x, cameraLookat.y, cameraLookat.z)
    val up = Vector[3](cameraUp.x, cameraUp.y, cameraUp.z)
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
        val lightDirection = Vector[3](-1f, 1f, -1f)  // Light from upper-left-back (Y positive = from above)
        val lightIntensity = 1.0f
        renderer.setLight(lightDirection, lightIntensity)
        logger.debug(s"Configured default light: direction=(${lightDirection(0)},${lightDirection(1)},${lightDirection(2)}), intensity=$lightIntensity")

  private def convertLightSpec(spec: LightSpec): Light =
    val position = Vector[3](spec.position.x, spec.position.y, spec.position.z)
    val color = spec.color.toCommonColor

    spec.lightType match
      case LightType.DIRECTIONAL => Light.Directional(position, color, spec.intensity)
      case LightType.POINT => Light.Point(position, color, spec.intensity)

  private def configurePlane(): Unit =
    val axisInt = planeSpec.axis match
      case Axis.X => 0
      case Axis.Y => 1
      case Axis.Z => 2
    renderer.setPlane(axisInt, planeSpec.positive, planeSpec.value)
    val axisName = planeSpec.axis.toString.toLowerCase
    val sign = if planeSpec.positive then "+" else "-"
    logger.debug(s"Configured plane: ${sign}${axisName}:${planeSpec.value}")

  def setSphereColor(color: Color): Unit =
    renderer.setSphereColor(color)
    logger.debug(s"Configured sphere color: RGBA=(${color.r}, ${color.g}, ${color.b}, ${color.a})")

  def setIOR(ior: Float): Unit =
    renderer.setIOR(ior)
    logger.debug(s"Configured index of refraction: IOR=$ior")

  def setScale(scale: Float): Unit =
    renderer.setScale(scale)
    logger.debug(s"Configured scale parameter: scale=$scale")

  def setShadows(enabled: Boolean): Unit =
    renderer.setShadows(enabled)
    logger.debug(s"Configured shadow rays: enabled=$enabled")

  def setAntialiasing(enabled: Boolean, maxDepth: Int, threshold: Float): Unit =
    renderer.setAntialiasing(enabled, maxDepth, threshold)
    logger.debug(s"Configured antialiasing: enabled=$enabled, maxDepth=$maxDepth, threshold=$threshold")

  def setPlaneColor(spec: PlaneColorSpec): Unit =
    val c1 = spec.color1
    spec.color2 match
      case Some(c2) =>
        renderer.setPlaneCheckerColors(c1, c2)
        logger.debug(f"Configured checkered plane colors: light=(${c1.r}%.2f, ${c1.g}%.2f, ${c1.b}%.2f), dark=(${c2.r}%.2f, ${c2.g}%.2f, ${c2.b}%.2f)")
      case None =>
        renderer.setPlaneSolidColor(c1)
        logger.debug(f"Configured solid plane color: RGB=(${c1.r}%.2f, ${c1.g}%.2f, ${c1.b}%.2f)")

  def renderScene(size: ImageSize): Array[Byte] =
    logger.debug(s"[OptiXResources] renderScene: rendering at ${size.width}x${size.height}")
    renderer.render(size).getOrElse:
      logger.error("OptiX rendering failed - returned None")
      Array.emptyByteArray

  def renderSceneWithStats(size: ImageSize): menger.optix.RenderResult =
    renderer.renderWithStats(size)

  def updateCamera(eye: Vector3, lookAt: Vector3, up: Vector3): Unit =
    val eyeVec = Vector[3](eye.x, eye.y, eye.z)
    val lookAtVec = Vector[3](lookAt.x, lookAt.y, lookAt.z)
    val upVec = Vector[3](up.x, up.y, up.z)
    val horizontalFov = 45f
    renderer.setCamera(eyeVec, lookAtVec, upVec, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Updated camera: eye=(${eyeVec(0)},${eyeVec(1)},${eyeVec(2)}), lookAt=(${lookAtVec(0)},${lookAtVec(1)},${lookAtVec(2)}), up=(${upVec(0)},${upVec(1)},${upVec(2)})")

  def updateCameraAspectRatio(size: ImageSize): Unit =
    val horizontalFov = 45f  // Fixed horizontal FOV in degrees (aspect-ratio independent)

    // Update cached image dimensions BEFORE calling setCamera
    renderer.updateImageDimensions(size)

    val eye = Vector[3](cameraPos.x, cameraPos.y, cameraPos.z)
    val lookAt = Vector[3](cameraLookat.x, cameraLookat.y, cameraLookat.z)
    val up = Vector[3](cameraUp.x, cameraUp.y, cameraUp.z)

    renderer.setCamera(eye, lookAt, up, horizontalFovDegrees = horizontalFov)

  def dispose(): Unit =
    _rendererRef.get().foreach { r =>
      logger.debug("Disposing OptiX renderer")
      r.dispose()
    }
