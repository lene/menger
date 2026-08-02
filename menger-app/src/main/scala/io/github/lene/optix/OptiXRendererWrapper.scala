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

  protected def initializeRenderer: OptiXRenderer =
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

  /** Renders a frame. `None` on failure — a failed render must not masquerade as a valid
    * (empty) frame. Robust to both the current pinned optix-jni (returns null on failure) and
    * the post-Sprint-35 optix-jni (throws): both map to `None`, and callers keep the last frame
    * / retry rather than displaying garbage. */
  def renderScene(size: ImageSize): Option[Array[Byte]] =
    logger.debug(s"[OptiXRendererWrapper] renderScene: rendering at ${size.width}x${size.height}")
    try
      val bytes = Option(renderer.render(size)).filter(_.nonEmpty)
      if bytes.isEmpty then logger.error("OptiX rendering failed - renderer returned null/empty")
      bytes
    catch
      case e: Exception =>
        logger.error(s"OptiX rendering failed: ${e.getMessage}", e)
        None

  def renderSceneWithStats(size: ImageSize): Option[RenderResult] =
    renderer.renderWithStats(size).toScala

  def dispose(): Unit =
    _rendererRef.get().foreach { r =>
      logger.debug("Disposing OptiX renderer")
      r.dispose()
    }

  override def close(): Unit = dispose()
