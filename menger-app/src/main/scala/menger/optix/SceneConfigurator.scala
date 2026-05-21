package menger.optix

import com.typesafe.scalalogging.LazyLogging
import menger.common.Color
import menger.common.Const
import menger.common.FogConfig
import menger.common.Light
import menger.common.Vector

class SceneConfigurator(
  cameraPos: Vector[3],
  cameraLookat: Vector[3],
  cameraUp: Vector[3],
  lights: Array[Light] = Array.empty
) extends LazyLogging:

  def configureCamera(renderer: OptiXRenderer): Unit =
    val horizontalFov = Const.Renderer.horizontalFov
    renderer.setCamera(cameraPos, cameraLookat, cameraUp, horizontalFovDegrees = horizontalFov)
    logger.debug(s"Configured camera: eye=(${cameraPos(0)},${cameraPos(1)},${cameraPos(2)}), lookAt=(${cameraLookat(0)},${cameraLookat(1)},${cameraLookat(2)}), up=(${cameraUp(0)},${cameraUp(1)},${cameraUp(2)}), horizontalFOV=$horizontalFov")

  def configureLights(renderer: OptiXRenderer): Unit =
    if lights.nonEmpty then
      renderer.setLights(lights)
      logger.debug(s"Configured ${lights.length} light(s) from specification")
    else
      // Default single directional light (backward compatibility)
      val lightDirection = Vector[3](-1f, 1f, -1f)  // Light from upper-left-back (Y positive = from above)
      val lightIntensity = 1.0f
      renderer.setLight(lightDirection, lightIntensity)
      logger.debug(s"Configured default light: direction=(${lightDirection(0)},${lightDirection(1)},${lightDirection(2)}), intensity=$lightIntensity")

  def setBackgroundColor(renderer: OptiXRenderer, color: Color): Unit =
    renderer.setBackgroundColor(color.r, color.g, color.b)
    logger.debug(f"Configured background color: RGB=(${color.r}%.2f, ${color.g}%.2f, ${color.b}%.2f)")

  def setFog(renderer: OptiXRenderer, fog: FogConfig): Unit =
    renderer.setFog(fog.density, fog.color.r, fog.color.g, fog.color.b)
    logger.debug(f"Configured fog: density=${fog.density}%.3f RGB=(${fog.color.r}%.2f, ${fog.color.g}%.2f, ${fog.color.b}%.2f)")

