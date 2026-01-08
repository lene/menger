package menger.optix

import java.util.concurrent.atomic.AtomicReference

import com.typesafe.scalalogging.LazyLogging
import menger.common.ImageSize

class OptiXRendererWrapper(maxInstances: Int = 64) extends LazyLogging:

  private val _rendererRef = new AtomicReference[Option[OptiXRenderer]](None)

  def renderer: OptiXRenderer =
    _rendererRef.get() match
      case Some(r) => r
      case None =>
        val r = initializeRenderer
        _rendererRef.set(Some(r))
        r

  private def initializeRenderer: OptiXRenderer =
    // CRITICAL: Force companion object initialization before creating instance
    // Without this, the companion object's static initializer (which loads the native library)
    // may not run until after we try to call native methods, causing UnsatisfiedLinkError
    if !OptiXRenderer.isLibraryLoaded then
      val msg = "OptiX native library failed to load"
      logger.error(msg)
      scala.sys.error(msg)

    val r = OptiXRenderer()
    r.initialize(maxInstances)
    r.ensureAvailable().get  // Throws on failure - caught by Main

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
