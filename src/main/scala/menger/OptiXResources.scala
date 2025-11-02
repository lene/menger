package menger

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.optix.OptiXRenderer

class OptiXResources(configureGeometry: Try[OptiXRenderer => Unit]) extends LazyLogging:

  private lazy val renderer: OptiXRenderer = initializeRenderer

  private def errorExit(message: String): Unit =
    logger.error(message)
    System.exit(1)

  private def initializeRenderer: OptiXRenderer = OptiXRenderer().ensureAvailable()

  def initialize(): Unit =
    logger.debug("Configuring OptiX scene")
    configureGeometry match
      case Success(config) => config(renderer)
      case Failure(exception) => errorExit(
        s"Invalid geometry configuration: ${exception.getMessage}"
      )
    createCamera(Vector3(0, 0, 3))
    createLights()

  private def createCamera(cameraPos: Vector3): Unit =
    val eye = Array(cameraPos.x, cameraPos.y, cameraPos.z)
    val lookAt = Array(0f, 0f, 0f)
    val up = Array(0f, 1f, 0f)
    val fov = 45f
    renderer.setCamera(eye, lookAt, up, fov)
    logger.debug(s"Configured camera: eye=${eye.mkString(",")}, lookAt=${lookAt.mkString(",")}, fov=$fov")

  private def createLights(): Unit =
    val lightDirection = Array(-1f, -1f, -1f)
    val lightIntensity = 1.0f
    renderer.setLight(lightDirection, lightIntensity)
    logger.debug(s"Configured light: direction=${lightDirection.mkString(",")}, intensity=$lightIntensity")

  def setSphereColor(r: Float, g: Float, b: Float, a: Float = 1.0f): Unit =
    renderer.setSphereColor(r, g, b, a)
    logger.debug(s"Configured sphere color: RGBA=($r, $g, $b, $a)")

  def setIOR(ior: Float): Unit =
    renderer.setIOR(ior)
    logger.debug(s"Configured index of refraction: IOR=$ior")

  def renderScene(width: Int, height: Int): Array[Byte] =
    renderer.render(width, height)

  def dispose(): Unit =
    logger.debug("Disposing OptiX renderer")
    renderer.dispose()
