package menger.geometry

import java.util.concurrent.atomic.AtomicLong

import io.github.lene.optix.MengerRenderer

final class VideoLoader(path: String) extends MengerVideoApi, AutoCloseable:
  require(Option(path).exists(_.nonEmpty), "Video path must not be empty")
  require(VideoLoader.isNativeLibraryLoaded, "Menger native library failed to load")

  private val nativeHandle = AtomicLong(openVideoNative(path))

  require(nativeHandle.get() != VideoLoader.ClosedHandle, s"Failed to open video: $path")

  def width: Int = videoWidthNative(openHandle)

  def height: Int = videoHeightNative(openHandle)

  def frameCount: Int = frameCountNative(openHandle)

  def durationSeconds: Double = videoDurationSecondsNative(openHandle)

  def nativeFps: Double = nativeFpsNative(openHandle)

  def frameAt(timestampSeconds: Double): Array[Byte] =
    getFrameAtNative(openHandle, timestampSeconds)

  def prefetch(timestampSeconds: Double, nFrames: Int): Unit =
    prefetchVideoNative(openHandle, timestampSeconds, nFrames)

  override def close(): Unit =
    val handle = nativeHandle.getAndSet(VideoLoader.ClosedHandle)
    if handle != VideoLoader.ClosedHandle then closeVideoNative(handle)

  private def openHandle: Long =
    val handle = nativeHandle.get()
    require(handle != VideoLoader.ClosedHandle, "Video loader is closed")
    handle

  @native override protected def openVideoNative(path: String): Long

  @native override protected def videoWidthNative(handle: Long): Int

  @native override protected def videoHeightNative(handle: Long): Int

  @native override protected def frameCountNative(handle: Long): Int

  @native override protected def videoDurationSecondsNative(handle: Long): Double

  @native override protected def nativeFpsNative(handle: Long): Double

  @native override protected def getFrameAtNative(
      handle: Long,
      timestampSeconds: Double
  ): Array[Byte]

  @native override protected def prefetchVideoNative(
      handle: Long,
      timestampSeconds: Double,
      nFrames: Int
  ): Unit

  @native override protected def closeVideoNative(handle: Long): Unit

object VideoLoader:
  private val ClosedHandle = 0L

  private def isNativeLibraryLoaded: Boolean = MengerRenderer.isLibraryLoaded
