package menger

import java.util.concurrent.atomic.AtomicReference

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.Axis
import menger.PlaneSpec
import menger.optix.OptiXRenderer

class OptiXResources(
  configureGeometry: Try[OptiXRenderer => Unit],
  cameraPos: Vector3,
  cameraLookat: Vector3,
  cameraUp: Vector3,
  planeSpec: PlaneSpec
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

  private def initializeRenderer: OptiXRenderer = OptiXRenderer().ensureAvailable()

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
    val fov = 45f
    renderer.setCamera(eye, lookAt, up, fov)
    logger.debug(s"Configured camera: eye=${eye.mkString(",")}, lookAt=${lookAt.mkString(",")}, up=${up.mkString(",")}, fov=$fov")

  private def createLights(): Unit =
    val lightDirection = Array(-1f, -1f, -1f)
    val lightIntensity = 1.0f
    renderer.setLight(lightDirection, lightIntensity)
    logger.debug(s"Configured light: direction=${lightDirection.mkString(",")}, intensity=$lightIntensity")

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

  def renderScene(width: Int, height: Int): Array[Byte] =
    renderer.render(width, height)

  def dispose(): Unit =
    _rendererRef.get().foreach { r =>
      logger.debug("Disposing OptiX renderer")
      r.dispose()
    }
