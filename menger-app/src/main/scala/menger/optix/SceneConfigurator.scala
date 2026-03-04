package menger.optix

import scala.util.Try

import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.ColorConversions.toCommonColor
import menger.Vector3Extensions.toVector3
import menger.cli.Axis
import menger.cli.LightSpec
import menger.cli.LightType
import menger.cli.PlaneConfig
import menger.common.Color
import menger.common.Const
import menger.common.Light
import menger.common.Vector

class SceneConfigurator(
  configureGeometry: Try[OptiXRenderer => Unit],
  cameraPos: Vector3,
  cameraLookat: Vector3,
  cameraUp: Vector3,
  lights: List[LightSpec] = List.empty
) extends LazyLogging:

  def configureScene(renderer: OptiXRenderer): Unit =
    logger.debug("Configuring OptiX scene")
    configureGeometry.get(renderer)  // Throws on failure - caught by Main
    configureCamera(renderer)
    configureLights(renderer)

  def configureCamera(renderer: OptiXRenderer): Unit =
    val eye = cameraPos.toVector3
    val lookAt = cameraLookat.toVector3
    val up = cameraUp.toVector3
    val horizontalFov = Const.Renderer.horizontalFov
    renderer.setCamera(eye, lookAt, up, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Configured camera: eye=(${eye(0)},${eye(1)},${eye(2)}), lookAt=(${lookAt(0)},${lookAt(1)},${lookAt(2)}), up=(${up(0)},${up(1)},${up(2)}), horizontalFOV=$horizontalFov")

  def configureLights(renderer: OptiXRenderer): Unit =
    if lights.nonEmpty then
      val lightSeq = lights.map(convertLightSpec)
      renderer.setLights(lightSeq)
      logger.debug(s"Configured ${lightSeq.length} light(s) from CLI specification")
    else
      // Default single directional light (backward compatibility)
      val lightDirection = Vector[3](-1f, 1f, -1f)  // Light from upper-left-back (Y positive = from above)
      val lightIntensity = 1.0f
      renderer.setLight(lightDirection, lightIntensity)
      logger.debug(s"Configured default light: direction=(${lightDirection(0)},${lightDirection(1)},${lightDirection(2)}), intensity=$lightIntensity")

  private def convertLightSpec(spec: LightSpec): Light =
    val position = spec.position.toVector3
    val color = spec.color.toCommonColor

    spec.lightType match
      case LightType.DIRECTIONAL => Light.Directional(position, color, spec.intensity)
      case LightType.POINT => Light.Point(position, color, spec.intensity)

  def configurePlanes(renderer: OptiXRenderer, planes: List[PlaneConfig]): Unit =
    renderer.clearPlanes()
    planes.foreach { planeConfig =>
      val axisInt = planeConfig.spec.axis match
        case Axis.X => 0
        case Axis.Y => 1
        case Axis.Z => 2
      planeConfig.colorSpec match
        case Some(colorSpec) =>
          colorSpec.color2 match
            case Some(c2) =>
              val c1 = colorSpec.color1
              renderer.addPlaneCheckerColors(
                axisInt, planeConfig.spec.positive, planeConfig.spec.value,
                c1.r, c1.g, c1.b, c2.r, c2.g, c2.b
              )
              logger.debug(f"Configured checkered plane: ${planeConfig.spec.axis}@${planeConfig.spec.value}")
            case None =>
              val c1 = colorSpec.color1
              renderer.addPlaneSolidColor(
                axisInt, planeConfig.spec.positive, planeConfig.spec.value,
                c1.r, c1.g, c1.b
              )
              logger.debug(f"Configured solid-color plane: ${planeConfig.spec.axis}@${planeConfig.spec.value}")
        case None =>
          renderer.addPlane(axisInt, planeConfig.spec.positive, planeConfig.spec.value)
          logger.debug(s"Configured default-color plane: ${planeConfig.spec.axis}@${planeConfig.spec.value}")
    }
    if planes.isEmpty then logger.debug("No planes configured")

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

  def setBackgroundColor(renderer: OptiXRenderer, color: Color): Unit =
    renderer.setBackgroundColor(color.r, color.g, color.b)
    logger.debug(f"Configured background color: RGB=(${color.r}%.2f, ${color.g}%.2f, ${color.b}%.2f)")

  def setShadows(renderer: OptiXRenderer, enabled: Boolean): Unit =
    renderer.setShadows(enabled)
    logger.debug(s"Configured shadow rays: enabled=$enabled")

  def setAntialiasing(renderer: OptiXRenderer, enabled: Boolean, maxDepth: Int, threshold: Float): Unit =
    renderer.setAntialiasing(enabled, maxDepth, threshold)
    logger.debug(s"Configured antialiasing: enabled=$enabled, maxDepth=$maxDepth, threshold=$threshold")

  def setCaustics(renderer: OptiXRenderer, enabled: Boolean, photonsPerIter: Int, iterations: Int, initialRadius: Float, alpha: Float): Unit =
    renderer.setCaustics(enabled, photonsPerIter, iterations, initialRadius, alpha)
    logger.debug(s"Configured caustics: enabled=$enabled, photonsPerIter=$photonsPerIter, iterations=$iterations, initialRadius=$initialRadius, alpha=$alpha")
