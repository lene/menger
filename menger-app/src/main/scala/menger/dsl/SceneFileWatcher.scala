package menger.dsl

import java.nio.file.ClosedWatchServiceException
import java.nio.file.FileSystems
import java.nio.file.Path
import java.nio.file.StandardWatchEventKinds
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

import com.typesafe.scalalogging.LazyLogging

/** Watches a single file for changes and invokes `onChange` after a short debounce window,
  * coalescing the burst of events one editor save often produces (a temp-file write followed
  * by a rename, or several small writes). Runs its own daemon threads; `close()` stops them.
  *
  * Used by `InteractiveEngine` for the interactive window's live scene reload (usability
  * review 2026-09, F5) -- kept free of any renderer/engine/LibGDX dependency so it is
  * unit-testable without a GPU context.
  */
class SceneFileWatcher(file: java.io.File, debounceMillis: Long = 200L)(onChange: () => Unit)
    extends LazyLogging:

  private val targetName = file.toPath.getFileName.toString
  private val watchService = FileSystems.getDefault.newWatchService()
  file.toPath.toAbsolutePath.getParent.register(
    watchService,
    StandardWatchEventKinds.ENTRY_MODIFY,
    StandardWatchEventKinds.ENTRY_CREATE
  )

  private val scheduler = Executors.newSingleThreadScheduledExecutor { r =>
    val t = new Thread(r, s"scene-file-watcher-debounce-$targetName")
    t.setDaemon(true)
    t
  }
  private val pendingDebounce = new AtomicReference[Option[ScheduledFuture[?]]](None)

  private val pollThread = new Thread(() => pollLoop(), s"scene-file-watcher-poll-$targetName")
  pollThread.setDaemon(true)
  pollThread.start()

  /** Blocks on `watchService.take()` until an event arrives or `close()` unblocks it with
    * `ClosedWatchServiceException`. Tail-recursive rather than a `while` loop (WartRemover
    * disables both `var` and `while` in production code). */
  @scala.annotation.tailrec
  private def pollLoop(): Unit =
    val keepGoing =
      try
        val key = watchService.take()
        val relevant = key.pollEvents().stream().anyMatch { ev =>
          ev.context() match
            case p: Path => p.getFileName.toString == targetName
            case _       => false
        }
        if relevant then scheduleDebounced()
        Some(key.reset())
      catch
        case _: ClosedWatchServiceException => None
        case _: InterruptedException        => None
    keepGoing match
      case Some(true) => pollLoop()
      case _          => ()

  private def scheduleDebounced(): Unit =
    pendingDebounce.get().foreach(_.cancel(false))
    val task = scheduler.schedule(
      new Runnable:
        def run(): Unit =
          try onChange()
          catch case e: Exception => logger.error(s"Scene file reload callback failed for $targetName", e)
      ,
      debounceMillis,
      TimeUnit.MILLISECONDS
    )
    pendingDebounce.set(Some(task))

  /** Stops watching. Idempotent: `watchService.close()` unblocks the poll thread's
    * `take()` with `ClosedWatchServiceException`, which ends `pollLoop`'s recursion. */
  def close(): Unit =
    watchService.close()
    scheduler.shutdownNow()
