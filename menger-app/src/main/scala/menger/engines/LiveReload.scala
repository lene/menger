package menger.engines

import java.io.File
import java.util.concurrent.atomic.AtomicReference

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import menger.ObjectSpec
import menger.dsl.LoadedScene
import menger.scene.SceneFileWatcher
import menger.scene.SceneLoader

/** The live reload both windows share (review 2026-10-14, menger#101: it existed twice and
  * the two copies already behaved differently). Watches the scene file, loads it with the total
  * loader, and checks the new scene before anything of the current one is torn down. The
  * window's life-cycle lines below are the ones the scene agent reads -- one wording for both
  * windows (the preview's "keeping the current one" never matched). */
trait LiveReload extends LazyLogging:
  self: BaseEngine =>

  private val fileWatcher = new AtomicReference[Option[SceneFileWatcher]](None)

  /** The scene file the window watches, when it was started on one: its events go next to it.
    * Known before `create()`, so a startup failure is reported too. */
  protected def eventsFor: Option[File]

  private lazy val events: Option[WindowEvents] =
    eventsFor.map(f => WindowEvents(WindowEvents.pathFor(f)))

  private def emit(kind: WindowEvents.Kind, detail: String): Unit =
    events.foreach(_.emit(kind, detail))

  /** Starts watching `file`; `onLoaded` runs on the watcher's thread with the loader's result. */
  protected def watchSceneFile(file: File)(onLoaded: Either[String, LoadedScene] => Unit): Unit =
    val watcher = new SceneFileWatcher(file)(() => onLoaded(SceneLoader.load(file.getPath)))
    fileWatcher.set(Some(watcher))
    logger.info(s"Watching scene file for live reload: ${file.getPath}")
    emit(WindowEvents.Kind.Up, "")

  /** The first scene could not be built: the window exits non-zero after this. */
  protected def startupFailed(cause: Throwable): Unit =
    emit(WindowEvents.Kind.StartupFailed, Option(cause.getMessage).getOrElse(cause.toString))

  /** A frame (or a rebuild) failed to build; the window keeps showing the last good one. */
  protected def frameFailed(context: String, cause: Throwable): Unit =
    logger.error(FrameBuildFailure.message(context, cause), cause)
    emit(WindowEvents.Kind.FrameFailed, FrameBuildFailure.message(context, cause)
      .stripPrefix(FrameBuildFailure.Marker + " "))

  protected def stopWatchingSceneFile(): Unit = fileWatcher.get().foreach(_.close())

  /** Everything that can refuse `specs` without touching the renderer: the scene budget and
    * the builders' own checks. A reload that fails here leaves the current scene untouched. */
  protected def precheck(specs: List[ObjectSpec]): Try[Unit] =
    withinBudget(specs).flatMap { _ =>
      BuildPrecheck.problems(specs)(using profilingConfig) match
        case Nil      => scala.util.Success(())
        case problems => scala.util.Failure(IllegalArgumentException(problems.mkString("; ")))
    }

  protected def reloaded(what: String, detail: String): Unit =
    logger.info(s"Reloaded $what from file ($detail)")
    emit(WindowEvents.Kind.Reloaded, detail)

  protected def reloadFailed(file: File, reason: String): Unit =
    logger.warn(s"Failed to reload ${file.getPath}, keeping the current scene: $reason")
    emit(WindowEvents.Kind.ReloadFailed, reason)

  protected def kindChanged(file: File, toAnimated: Boolean): Unit =
    val (now, window) = if toAnimated then ("an animated", "this static window")
                        else ("a static", "this animated window")
    val message = s"${file.getPath} changed to $now scene, which $window cannot show -- " +
      "restart the window to pick it up"
    logger.warn(message)
    emit(WindowEvents.Kind.KindChanged, message)
