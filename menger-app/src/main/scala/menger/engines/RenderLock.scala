package menger.engines

import java.io.RandomAccessFile
import java.nio.channels.FileChannel
import java.nio.channels.FileLock
import java.nio.channels.OverlappingFileLockException

import scala.util.control.NonFatal

import com.typesafe.scalalogging.LazyLogging

/** GPU-exclusivity guard (AD-16: "at most one active render session ... a second request ...
  * refused explicitly, never queued silently") built on `java.nio.channels.FileLock`.
  *
  * `tryAcquire` is non-blocking by construction (`FileChannel.tryLock()`, never `lock()`):
  * a path already held -- by this JVM or another process -- fails immediately with a `Left`
  * rather than waiting or retrying. The returned `Handle` ties the OS-level lock to an open
  * `FileChannel`; closing the handle releases the lock explicitly, and if a caller never
  * gets the chance to (a crash, `sys.exit` skipping cleanup), the OS itself releases file
  * locks when the holding process's file descriptors close at JVM exit -- either path leaves
  * the lock free for the next attempt.
  */
object RenderLock extends LazyLogging:

  /** A held render lock. `close()` releases it and frees `path` for a subsequent
    * `tryAcquire` -- safe to call more than once. */
  final class Handle private[RenderLock] (channel: FileChannel, lock: FileLock):
    def close(): Unit =
      try
        if lock.isValid then lock.release()
      finally
        if channel.isOpen then channel.close()

  /** Attempts to acquire an exclusive lock on `path`. Never blocks: a lock already held --
    * by another process (`tryLock()` returns `null`) or by this same JVM on an overlapping
    * region (`OverlappingFileLockException`) -- is reported as a `Left`, not a wait/retry.
    * A missing parent directory (or any other I/O failure opening the lock file) is also
    * reported as a `Left` naming `path`, never silently redirected to a fallback location. */
  def tryAcquire(path: String): Either[String, Handle] =
    try
      val raf = new RandomAccessFile(path, "rw")
      val channel = raf.getChannel
      try
        Option(channel.tryLock()) match
          case None =>
            channel.close()
            Left(s"render lock already held: $path")
          case Some(lock) =>
            Right(Handle(channel, lock))
      catch
        case _: OverlappingFileLockException =>
          channel.close()
          Left(s"render lock already held: $path")
        // Review round 2: only OverlappingFileLockException was caught here, so any other
        // failure from tryLock() (an IOException on a full or read-only filesystem, an NFS
        // mount without lock support) propagated to the outer handler with `raf` and
        // `channel` still open -- a file-descriptor leak on every such attempt.
        case NonFatal(e) =>
          channel.close()
          logger.error(s"Failed to lock '$path'", e)
          Left(s"failed to acquire render lock at '$path': ${e.getMessage}")
    catch
      case NonFatal(e) =>
        logger.error(s"Failed to acquire render lock at '$path'", e)
        Left(s"failed to acquire render lock at '$path': ${e.getMessage}")
