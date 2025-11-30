package menger.optix

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
import menger.common.Light
import menger.common.Vector

class SceneConfigurator(
  configureGeometry: Try[OptiXRenderer => Unit],
  cameraPos: Vector3,
  cameraLookat: Vector3,
  cameraUp: Vector3,
  planeSpec: PlaneSpec,
  lights: Option[List[LightSpec]] = None
) extends LazyLogging:

  def configureScene(renderer: OptiXRenderer): Unit =
    logger.debug("Configuring OptiX scene")
    configureGeometry.get(renderer)  // Throws on failure - caught by Main
    createCamera(renderer)
    createLights(renderer)
    configurePlane(renderer)

  private def createCamera(renderer: OptiXRenderer): Unit =
    val eye = Vector[3](cameraPos.x, cameraPos.y, cameraPos.z)
    val lookAt = Vector[3](cameraLookat.x, cameraLookat.y, cameraLookat.z)
    val up = Vector[3](cameraUp.x, cameraUp.y, cameraUp.z)
    val horizontalFov = 45f
    renderer.setCamera(eye, lookAt, up, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Configured camera: eye=(${eye(0)},${eye(1)},${eye(2)}), lookAt=(${lookAt(0)},${lookAt(1)},${lookAt(2)}), up=(${up(0)},${up(1)},${up(2)}), horizontalFOV=$horizontalFov")

  private def createLights(renderer: OptiXRenderer): Unit =
    lights match
      case Some(lightSpecs) =>
        val lightSeq = lightSpecs.map(convertLightSpec)
        renderer.setLights(lightSeq)  // Throws on failure - caught by Main
        logger.debug(s"Configured ${lightSeq.length} light(s) from CLI specification")
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

  private def configurePlane(renderer: OptiXRenderer): Unit =
    val axisInt = planeSpec.axis match
      case Axis.X => 0
      case Axis.Y => 1
      case Axis.Z => 2
    renderer.setPlane(axisInt, planeSpec.positive, planeSpec.value)
    val axisName = planeSpec.axis.toString.toLowerCase
    val sign = if planeSpec.positive then "+" else "-"
    logger.debug(s"Configured plane: ${sign}${axisName}:${planeSpec.value}")

  def setSphereColor(renderer: OptiXRenderer, color: Color): Unit =
    renderer.setSphereColor(color)
    logger.debug(s"Configured sphere color: RGBA=(${color.r}, ${color.g}, ${color.b}, ${color.a})")

  def setTriangleMeshColor(renderer: OptiXRenderer, color: Color): Unit =
    renderer.setTriangleMeshColor(color)
    logger.debug(s"Configured triangle mesh color: RGBA=(${color.r}, ${color.g}, ${color.b}, ${color.a})")

  def setTriangleMeshIOR(renderer: OptiXRenderer, ior: Float): Unit =
    renderer.setTriangleMeshIOR(ior)
    logger.debug(s"Configured triangle mesh IOR: $ior")

  def setIOR(renderer: OptiXRenderer, ior: Float): Unit =
    renderer.setIOR(ior)
    logger.debug(s"Configured index of refraction: IOR=$ior")

  def setScale(renderer: OptiXRenderer, scale: Float): Unit =
    renderer.setScale(scale)
    logger.debug(s"Configured scale parameter: scale=$scale")

  def setShadows(renderer: OptiXRenderer, enabled: Boolean): Unit =
    renderer.setShadows(enabled)
    logger.debug(s"Configured shadow rays: enabled=$enabled")

  def setAntialiasing(renderer: OptiXRenderer, enabled: Boolean, maxDepth: Int, threshold: Float): Unit =
    renderer.setAntialiasing(enabled, maxDepth, threshold)
    logger.debug(s"Configured antialiasing: enabled=$enabled, maxDepth=$maxDepth, threshold=$threshold")

  def setCaustics(renderer: OptiXRenderer, enabled: Boolean, photonsPerIter: Int, iterations: Int, initialRadius: Float, alpha: Float): Unit =
    renderer.setCaustics(enabled, photonsPerIter, iterations, initialRadius, alpha)
    logger.debug(s"Configured caustics: enabled=$enabled, photonsPerIter=$photonsPerIter, iterations=$iterations, initialRadius=$initialRadius, alpha=$alpha")

  def setPlaneColor(renderer: OptiXRenderer, spec: PlaneColorSpec): Unit =
    val c1 = spec.color1
    spec.color2 match
      case Some(c2) =>
        renderer.setPlaneCheckerColors(c1, c2)
        logger.debug(f"Configured checkered plane colors: light=(${c1.r}%.2f, ${c1.g}%.2f, ${c1.b}%.2f), dark=(${c2.r}%.2f, ${c2.g}%.2f, ${c2.b}%.2f)")
      case None =>
        renderer.setPlaneSolidColor(c1)
        logger.debug(f"Configured solid plane color: RGB=(${c1.r}%.2f, ${c1.g}%.2f, ${c1.b}%.2f)")
