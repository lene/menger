package menger.engines

import java.nio.file.Files
import java.nio.file.Paths
import java.util.concurrent.atomic.AtomicReference

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.OptiXRendererWrapper
import io.github.lene.optix.RenderResult
import menger.common.ImageSize

/** Render-stats concern extracted from InteractiveEngine (F13, Sprint 35 Ph4).
  *
  * Mix into an engine that needs optional per-frame stats logging + JSON output.
  * The engine provides `enableStats`, `statsJsonPath`, and the `rendererWrapper`;
  * the trait handles the render-path branch (plain vs stats), the log line, the
  * JSON formatting, and writing the file on dispose.
  */
trait WithStats extends LazyLogging:

  /** Whether to collect and log per-frame render stats. */
  protected def enableStats: Boolean

  /** Optional path to write a stats JSON file on dispose. */
  protected def statsJsonPath: Option[String]

  /** The renderer wrapper used for both plain and stats render calls. */
  protected def rendererWrapper: OptiXRendererWrapper

  private[engines] val lastRenderResult = new AtomicReference[Option[RenderResult]](None)

  /** Renders one frame, collecting stats if `enableStats` is true. */
  private[engines] def maybeRenderWithStats(width: Int, height: Int): Option[Array[Byte]] =
    if !enableStats then
      rendererWrapper.renderScene(ImageSize(width, height))
    else
      rendererWrapper.renderSceneWithStats(ImageSize(width, height)) match
        case None =>
          logger.error("OptiX rendering failed - renderWithStats returned None")
          None
        case Some(result) =>
          lastRenderResult.set(Some(result))
          logStats(result)
          Some(result.image)

  /** Writes the pending stats JSON (if any) to `statsJsonPath`. Call from dispose. */
  protected def disposeStats(): Unit =
    statsJsonPath.foreach(writeStatsJson)

  private def logStats(result: RenderResult): Unit =
    val stats = result.stats
    logger.info(
      f"Frame: ${stats.frameMs}%.1f ms (${stats.msPerMray}%.2f ms/Mray) | " +
      s"primary=${stats.primaryRays} total=${stats.totalRays} " +
      s"reflected=${stats.reflectedRays} refracted=${stats.refractedRays} " +
      s"shadow=${stats.shadowRays} aa=${stats.aaRays} spectral=${stats.spectralRays} " +
      s"depth=${stats.minDepthReached}-${stats.maxDepthReached}"
    )

  /** Formats the last render result as a JSON string. Returns None if no result. */
  private[engines] def statsJson: Option[String] =
    lastRenderResult.get().map { result =>
      val s = result.stats
      s"""|{
          |  "frameMs": ${s.frameMs},
          |  "totalRays": ${s.totalRays},
          |  "primaryRays": ${s.primaryRays},
          |  "reflectedRays": ${s.reflectedRays},
          |  "refractedRays": ${s.refractedRays},
          |  "shadowRays": ${s.shadowRays},
          |  "aaRays": ${s.aaRays},
          |  "spectralRays": ${s.spectralRays},
          |  "msPerMray": ${s.msPerMray}
          |}""".stripMargin
    }

  private def writeStatsJson(path: String): Unit =
    statsJson match
      case None =>
        logger.warn(s"No render result available; stats file not written: $path")
      case Some(json) =>
        Try {
          val p = Paths.get(path).toAbsolutePath
          Option(p.getParent).foreach(parent => Files.createDirectories(parent))
          Files.writeString(p, json)
          logger.info(s"Stats written to $p")
        }.failed.foreach { e =>
          logger.error(s"Failed to write stats to $path: ${e.getMessage}", e)
        }
