package io.github.lene.optix

import java.util.concurrent.atomic.AtomicReference

import scala.jdk.OptionConverters._

import com.typesafe.scalalogging.LazyLogging
import menger.common.ImageSize

class OptiXRendererWrapper(maxInstances: Int = 64) extends LazyLogging with AutoCloseable:

  private val _rendererRef = new AtomicReference[Option[OptiXRenderer]](None)

  def renderer: OptiXRenderer =
    _rendererRef.get() match
      case Some(r) => r
      case None =>
        val r = initializeRenderer
        _rendererRef.set(Some(r))
        r

  private def initializeRenderer: OptiXRenderer =
    if !OptiXRenderer.isLibraryLoaded then
      val msg = "OptiX native library failed to load"
      logger.error(msg)
      scala.sys.error(msg)
    if !MengerRenderer.isLibraryLoaded then
      val msg = "Menger native library failed to load"
      logger.error(msg)
      scala.sys.error(msg)

    val r = MengerRenderer()
    r.initialize(maxInstances)
    r.ensureAvailable()  // Throws OptiXNotAvailableException on failure - caught by Main

  def renderScene(size: ImageSize): Array[Byte] =
    logger.debug(s"[OptiXRendererWrapper] renderScene: rendering at ${size.width}x${size.height}")
    val result = renderer.render(size)
    if result != null then result // scalafix:ok DisableSyntax.null
    else
      logger.error("OptiX rendering failed - returned null")
      Array.emptyByteArray

  def renderSceneWithStats(size: ImageSize): Option[RenderResult] =
    renderer.renderWithStats(size).toScala

  def dispose(): Unit =
    _rendererRef.get().foreach { r =>
      logger.debug("Disposing OptiX renderer")
      r.dispose()
    }

  override def close(): Unit = dispose()
