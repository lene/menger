package menger.optix

import com.typesafe.scalalogging.LazyLogging
import menger.cli.Axis
import menger.cli.PlaneConfig
import menger.common.Color
import menger.common.Const
import menger.common.FogConfig
import menger.common.Light
import menger.common.Vector

class SceneConfigurator(
  cameraPos: Vector[3],
  cameraLookat: Vector[3],
  cameraUp: Vector[3],
  lights: List[Light] = List.empty
) extends LazyLogging:

  def configureCamera(renderer: OptiXRenderer): Unit =
    val horizontalFov = Const.Renderer.horizontalFov
    renderer.setCamera(cameraPos, cameraLookat, cameraUp, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Configured camera: eye=(${cameraPos(0)},${cameraPos(1)},${cameraPos(2)}), lookAt=(${cameraLookat(0)},${cameraLookat(1)},${cameraLookat(2)}), up=(${cameraUp(0)},${cameraUp(1)},${cameraUp(2)}), horizontalFOV=$horizontalFov")

  def configureLights(renderer: OptiXRenderer): Unit =
    if lights.nonEmpty then
      renderer.setLights(lights.toArray)
      logger.debug(s"Configured ${lights.length} light(s) from specification")
    else
      // Default single directional light (backward compatibility)
      val lightDirection = Vector[3](-1f, 1f, -1f)  // Light from upper-left-back (Y positive = from above)
      val lightIntensity = 1.0f
      renderer.setLight(lightDirection, lightIntensity)
      logger.debug(s"Configured default light: direction=(${lightDirection(0)},${lightDirection(1)},${lightDirection(2)}), intensity=$lightIntensity")

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
              planeConfig.material match
                case Some(mat) =>
                  renderer.addPlaneCheckerColorsWithMaterial(
                    axisInt, planeConfig.spec.positive, planeConfig.spec.value,
                    Color(c1.r, c1.g, c1.b, 1.0f), Color(c2.r, c2.g, c2.b, 1.0f), mat
                  )
                case None =>
                  renderer.addPlaneCheckerColors(
                    axisInt, planeConfig.spec.positive, planeConfig.spec.value,
                    c1.r, c1.g, c1.b, c2.r, c2.g, c2.b
                  )
              logger.debug(f"Configured checkered plane: ${planeConfig.spec.axis}@${planeConfig.spec.value}")
            case None =>
              val c1 = colorSpec.color1
              planeConfig.material match
                case Some(mat) =>
                  renderer.addPlaneSolidColorWithMaterial(
                    axisInt, planeConfig.spec.positive, planeConfig.spec.value,
                    Color(c1.r, c1.g, c1.b, 1.0f), mat
                  )
                case None =>
                  renderer.addPlaneSolidColor(
                    axisInt, planeConfig.spec.positive, planeConfig.spec.value,
                    c1.r, c1.g, c1.b
                  )
              logger.debug(f"Configured solid-color plane: ${planeConfig.spec.axis}@${planeConfig.spec.value}")
        case None =>
          planeConfig.material match
            case Some(mat) =>
              // No explicit colour: use the material's own colour as a solid floor.
              // Checker pattern is opt-in via --plane-color RRGGBB:RRGGBB.
              renderer.addPlaneSolidColorWithMaterial(
                axisInt, planeConfig.spec.positive, planeConfig.spec.value,
                mat.color, mat
              )
            case None =>
              renderer.addPlane(axisInt, planeConfig.spec.positive, planeConfig.spec.value)
          logger.debug(s"Configured default-color plane: ${planeConfig.spec.axis}@${planeConfig.spec.value}")
    }
    if planes.isEmpty then logger.debug("No planes configured")

  def setBackgroundColor(renderer: OptiXRenderer, color: Color): Unit =
    renderer.setBackgroundColor(color.r, color.g, color.b)
    logger.debug(f"Configured background color: RGB=(${color.r}%.2f, ${color.g}%.2f, ${color.b}%.2f)")

  def setFog(renderer: OptiXRenderer, fog: FogConfig): Unit =
    renderer.setFog(fog.density, fog.color.r, fog.color.g, fog.color.b)
    logger.debug(f"Configured fog: density=${fog.density}%.3f RGB=(${fog.color.r}%.2f, ${fog.color.g}%.2f, ${fog.color.b}%.2f)")

