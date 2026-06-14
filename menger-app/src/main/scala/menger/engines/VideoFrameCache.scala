package menger.engines

import scala.collection.mutable

private[engines] final class VideoFrameCache(maxFrames: Int):
  require(maxFrames > 0, "Video frame cache size must be positive")

  private val frames = mutable.LinkedHashMap.empty[Double, Array[Byte]]

  def getOrDecode(timestampSeconds: Double)(decode: => Array[Byte]): Array[Byte] =
    frames.remove(timestampSeconds) match
      case Some(frame) =>
        frames.update(timestampSeconds, frame)
        frame
      case None =>
        val frame = decode
        frames.update(timestampSeconds, frame)
        trimToLimit()
        frame

  def size: Int = frames.size

  def contains(timestampSeconds: Double): Boolean = frames.contains(timestampSeconds)

  private def trimToLimit(): Unit =
    if frames.size > maxFrames then
      frames.remove(frames.head._1)
      trimToLimit()

private[engines] object VideoFrameCache:
  val DefaultMaxFrames = 8
