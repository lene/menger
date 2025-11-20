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
import menger.optix.Light
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
    val eye = Array(cameraPos.x, cameraPos.y, cameraPos.z)
    val lookAt = Array(cameraLookat.x, cameraLookat.y, cameraLookat.z)
    val up = Array(cameraUp.x, cameraUp.y, cameraUp.z)
    val horizontalFov = 45f
    renderer.setCamera(eye, lookAt, up, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Configured camera: eye=${eye.mkString(",")}, lookAt=${lookAt.mkString(",")}, up=${up.mkString(",")}, horizontalFOV=$horizontalFov")

  private def createLights(): Unit =
    lights match
      case Some(lightSpecs) =>
        val lightArray = lightSpecs.map(convertLightSpec).toArray
        Try(renderer.setLights(lightArray)) match
          case Success(_) =>
            logger.debug(s"Configured ${lightArray.length} light(s) from CLI specification")
          case Failure(exception) =>
            errorExit(s"Failed to configure lights: ${exception.getMessage}")
      case None =>
        // Default single directional light (backward compatibility)
        val lightDirection = Array(-1f, 1f, -1f)  // Light from upper-left-back (Y positive = from above)
        val lightIntensity = 1.0f
        renderer.setLight(lightDirection, lightIntensity)
        logger.debug(s"Configured default light: direction=${lightDirection.mkString(",")}, intensity=$lightIntensity")

  private def convertLightSpec(spec: LightSpec): Light =
    val lightType = spec.lightType match
      case LightType.DIRECTIONAL => menger.optix.LightType.DIRECTIONAL
      case LightType.POINT => menger.optix.LightType.POINT

    val position = Array(spec.position.x, spec.position.y, spec.position.z)
    val color = Array(spec.color.r, spec.color.g, spec.color.b)

    Light(
      lightType = lightType,
      direction = position,  // For directional lights, position is treated as direction
      position = position,   // For point lights
      color = color,
      intensity = spec.intensity
    )

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
    val eyeArr = Array(eye.x, eye.y, eye.z)
    val lookAtArr = Array(lookAt.x, lookAt.y, lookAt.z)
    val upArr = Array(up.x, up.y, up.z)
    val horizontalFov = 45f
    renderer.setCamera(eyeArr, lookAtArr, upArr, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Updated camera: eye=${eyeArr.mkString(",")}, lookAt=${lookAtArr.mkString(",")}, up=${upArr.mkString(",")}")

  def updateCameraAspectRatio(size: ImageSize): Unit =
    val horizontalFov = 45f  // Fixed horizontal FOV in degrees (aspect-ratio independent)

    // Update cached image dimensions BEFORE calling setCamera
    renderer.updateImageDimensions(size)

    val eye = Array(cameraPos.x, cameraPos.y, cameraPos.z)
    val lookAt = Array(cameraLookat.x, cameraLookat.y, cameraLookat.z)
    val up = Array(cameraUp.x, cameraUp.y, cameraUp.z)

    renderer.setCamera(eye, lookAt, up, horizontalFovDegrees = horizontalFov)

  def dispose(): Unit =
    _rendererRef.get().foreach { r =>
      logger.debug("Disposing OptiX renderer")
      r.dispose()
    }
