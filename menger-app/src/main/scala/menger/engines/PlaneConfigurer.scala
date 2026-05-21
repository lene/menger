package menger.engines

import com.typesafe.scalalogging.LazyLogging
import menger.common.Axis
import menger.common.Color
import menger.config.PlaneConfig
import menger.optix.OptiXRenderer

/** Applies plane configurations to an OptiX renderer.
  *
  * Extracted from SceneConfigurator to break the optix→cli dependency.
  * Lives in menger.engines, which may bridge cli and optix layers.
  */
object PlaneConfigurer extends LazyLogging:

  def configurePlanes(renderer: OptiXRenderer, planes: Array[PlaneConfig]): Unit =
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
