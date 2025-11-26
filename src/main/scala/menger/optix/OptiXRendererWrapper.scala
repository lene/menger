package menger.optix

import java.util.concurrent.atomic.AtomicReference

import com.typesafe.scalalogging.LazyLogging
import menger.common.ImageSize

/** Wrapper for OptiX JNI renderer handling initialization and rendering calls.
  *
  * Responsible for:
  * - JNI renderer lifecycle (initialization, disposal)
  * - Rendering scene to image buffers
  * - Rendering with statistics collection
  */
class OptiXRendererWrapper extends LazyLogging:

  private val _rendererRef = new AtomicReference[Option[OptiXRenderer]](None)

  def renderer: OptiXRenderer =
    _rendererRef.get() match
      case Some(r) => r
      case None =>
        val r = initializeRenderer
        _rendererRef.set(Some(r))
        r

  private def initializeRenderer: OptiXRenderer =
    OptiXRenderer().ensureAvailable().get  // Throws on failure - caught by Main

  def renderScene(size: ImageSize): Array[Byte] =
    logger.debug(s"[OptiXRendererWrapper] renderScene: rendering at ${size.width}x${size.height}")
    renderer.render(size).getOrElse:
      logger.error("OptiX rendering failed - returned None")
      Array.emptyByteArray

  def renderSceneWithStats(size: ImageSize): RenderResult =
    renderer.renderWithStats(size)

  def dispose(): Unit =
    _rendererRef.get().foreach { r =>
      logger.debug("Disposing OptiX renderer")
      r.dispose()
    }
