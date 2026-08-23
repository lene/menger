package menger.engines

import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.OptiXRenderer
import menger.common.Axis
import menger.common.Color
import menger.common.Material
import menger.common.Vector
import menger.config.PlaneConfig

/** Applies plane configurations to an OptiX renderer.
  *
  * Extracted from SceneConfigurator to break the optix→cli dependency.
  * Lives in menger.engines, which may bridge cli and optix layers.
  *
  * Uses `addPlaneInstance` (real BVH geometry, `hit_plane.cu`) rather than the
  * legacy miss-shader plane path (`--plane`'s original implementation): the
  * legacy path never enters ray-object intersection, so it could never occlude
  * real geometry on the far side, at any camera angle (Sprint 36 H3.1).
  */
object PlaneConfigurer extends LazyLogging:

  // Matches native RenderConfig::addPlane's default checker
  // (RayTracingConstants::PLANE_CHECKER_LIGHT_GRAY / DARK_GRAY, /255).
  private val DefaultLight = Color(120f / 255f, 120f / 255f, 120f / 255f, 1.0f)
  private val DefaultDark  = Color(20f / 255f, 20f / 255f, 20f / 255f, 1.0f)

  def configurePlanes(renderer: OptiXRenderer, planes: Array[PlaneConfig]): Unit =
    planes.foreach { planeConfig =>
      val sign = if planeConfig.spec.positive then 1f else -1f
      val normal = planeConfig.spec.axis match
        case Axis.X => Vector[3](sign, 0f, 0f)
        case Axis.Y => Vector[3](0f, sign, 0f)
        case Axis.Z => Vector[3](0f, 0f, sign)
      val distance = planeConfig.spec.value * sign

      planeConfig.colorSpec match
        case Some(colorSpec) =>
          val material = planeConfig.material
            .getOrElse(Material.matte(colorSpec.color1))
            .copy(color = colorSpec.color1)
          renderer.addPlaneInstance(normal, distance, material, colorSpec.color2.orNull)
        case None =>
          planeConfig.material match
            case Some(mat) =>
              // No explicit colour: use the material's own colour as a solid floor.
              // Checker pattern is opt-in via --plane-color RRGGBB:RRGGBB.
              renderer.addPlaneInstance(normal, distance, mat, null) // scalafix:ok DisableSyntax.null
            case None =>
              renderer.addPlaneInstance(normal, distance, Material.matte(DefaultLight), DefaultDark)
      logger.debug(s"Configured plane geometry: ${planeConfig.spec.axis}@${planeConfig.spec.value}")
    }
    if planes.isEmpty then logger.debug("No planes configured")
