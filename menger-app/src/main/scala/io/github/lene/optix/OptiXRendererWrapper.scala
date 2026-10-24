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
    * / retry rather than displaying garbage. A sticky CUDA error is the exception: the context
    * is dead for the rest of the process, so retrying only repeats the failure every frame —
    * it is raised as a fatal error instead. */
  def renderScene(size: ImageSize): Option[Array[Byte]] =
    logger.debug(s"[OptiXRendererWrapper] renderScene: rendering at ${size.width}x${size.height}")
    try
      val bytes = Option(renderer.render(size)).filter(_.nonEmpty)
      if bytes.isEmpty then logger.error("OptiX rendering failed - renderer returned null/empty")
      bytes
    catch
      case e: Exception if OptiXRendererWrapper.isStickyCudaError(e.getMessage) =>
        sys.error(
          s"OptiX rendering hit an unrecoverable CUDA error, restart required: ${e.getMessage}"
        )
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

object OptiXRendererWrapper:
  // The CUDA runtime documents these as leaving the process in an inconsistent state: "any
  // further CUDA work will return the same error" until the process is relaunched.
  private val StickyCudaErrorCodes = Set(700, 702, 714, 715, 716, 717, 718, 719)

  // optix-jni's CUDA_CHECK format: "CUDA call '<call>' failed: <description> (<code>)", with
  // extra explanation lines appended after the code for some errors (e.g. 718).
  private val CudaCheckFailure = """CUDA call '[^']*' failed: [^\n]*\((\d+)\)""".r

  def isStickyCudaError(message: String): Boolean =
    Option(message)
      .flatMap(CudaCheckFailure.findFirstMatchIn)
      .exists(m => StickyCudaErrorCodes.contains(m.group(1).toInt))
