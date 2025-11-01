package menger

import com.typesafe.scalalogging.LazyLogging
import menger.optix.OptiXRenderer

@SuppressWarnings(Array("org.wartremover.warts.Throw"))
class OptiXResources(configureGeometry: OptiXRenderer => Unit) extends LazyLogging:

  private lazy val renderer: OptiXRenderer = initializeRenderer

  private def errorExit(message: String): Unit =
    logger.error(message)
    System.exit(1)

  private def initializeRenderer: OptiXRenderer =
    if !OptiXRenderer.isLibraryLoaded then
      errorExit("OptiX native library failed to load - ensure CUDA and OptiX are available")

    val r = OptiXRenderer()
    if !r.isAvailable then
      errorExit("OptiX not available on this system - ensure CUDA and OptiX are available")
    if !r.initialize() then
      errorExit("Failed to initialize OptiX renderer")

    r

  def initialize(): Unit =
    logger.info("Configuring OptiX scene")
    configureGeometry(renderer)

    val eye = Array(0f, 0f, 3f)
    val lookAt = Array(0f, 0f, 0f)
    val up = Array(0f, 1f, 0f)
    val fov = 45f
    renderer.setCamera(eye, lookAt, up, fov)
    logger.debug(s"Configured camera: eye=${eye.mkString(",")}, lookAt=${lookAt.mkString(",")}, fov=$fov")

    val lightDirection = Array(-1f, -1f, -1f)
    val lightIntensity = 1.0f
    renderer.setLight(lightDirection, lightIntensity)
    logger.debug(s"Configured light: direction=${lightDirection.mkString(",")}, intensity=$lightIntensity")

  def renderScene(width: Int, height: Int): Array[Byte] =
    renderer.render(width, height)

  def dispose(): Unit =
    logger.debug("Disposing OptiX renderer")
    renderer.dispose()
