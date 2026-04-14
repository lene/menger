package menger.engines

import java.nio.file.Files
import java.nio.file.Paths

import com.typesafe.scalalogging.LazyLogging

trait WithVideoExport extends RenderEngine with LazyLogging:
  self: WithAnimation =>

  protected def videoOutputPath: String
  protected def videoQuality: Int
  protected def keepFrames: Boolean
  protected def videoFps: Int = 24

  override protected def onAnimationComplete(): Unit =
    super.onAnimationComplete()
    VideoEncoder.encode(animConfig.savePattern, videoOutputPath, videoFps, videoQuality)
    if !keepFrames then cleanupFrames()

  private def cleanupFrames(): Unit =
    val pattern = animConfig.savePattern
    val lastSlash = pattern.lastIndexOf('/')
    val (dir, filePrefix) =
      if lastSlash >= 0 then
        (pattern.substring(0, lastSlash), pattern.substring(lastSlash + 1))
      else
        (".", pattern)
    val prefixBeforePercent = filePrefix.takeWhile(_ != '%')
    val dirPath = Paths.get(dir)
    if Files.isDirectory(dirPath) then
      val stream = Files.list(dirPath)
      try
        stream
          .filter(p => p.getFileName.toString.startsWith(prefixBeforePercent))
          .filter(p => p.getFileName.toString.endsWith(".png"))
          .forEach { p =>
            logger.debug(s"Deleting frame file: $p")
            Files.deleteIfExists(p)
          }
      finally stream.close()
