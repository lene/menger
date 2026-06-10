package menger.geometry

trait MengerVideoApi:
  protected def openVideoNative(path: String): Long

  protected def videoWidthNative(handle: Long): Int

  protected def videoHeightNative(handle: Long): Int

  protected def frameCountNative(handle: Long): Int

  protected def videoDurationSecondsNative(handle: Long): Double

  protected def nativeFpsNative(handle: Long): Double

  protected def getFrameAtNative(handle: Long, timestampSeconds: Double): Array[Byte]

  protected def prefetchVideoNative(handle: Long, timestampSeconds: Double, nFrames: Int): Unit

  protected def closeVideoNative(handle: Long): Unit
