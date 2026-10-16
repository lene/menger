package menger.engines

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.StandardOpenOption
import java.time.Instant

import scala.util.Try

import upickle.default.ReadWriter
import upickle.default.write

/** The window's life cycle for the scene agent, one JSON object per line in
  * `<scene file>.events.jsonl` (contracts/process-result.json `windowEvents`). The agent used
  * to infer it from seven English substrings of the window's log, which a log-message cleanup or
  * `--log-level warn` broke without any test noticing (review 2026-10-14, menger#82). */
object WindowEvents:

  val Version = 1

  enum Kind(val wire: String):
    case Up            extends Kind("up")
    case Reloaded      extends Kind("reloaded")
    case ReloadFailed  extends Kind("reload_failed")
    case KindChanged   extends Kind("kind_changed")
    case FrameFailed   extends Kind("frame_failed")
    case StartupFailed extends Kind("startup_failed")

  final case class Event(v: Int, kind: String, detail: String, time: String) derives ReadWriter

  def pathFor(sceneFile: File): File = new File(sceneFile.getPath + ".events.jsonl")

  def line(kind: Kind, detail: String, now: Instant = Instant.now()): String =
    write(Event(Version, kind.wire, detail, now.toString))

/** Appends events to one file. Never throws: a failed write only loses that event, the window
  * goes on. Truncated when created, so one launch's events are one file. */
final class WindowEvents(target: File):
  Try(Files.write(target.toPath, Array.emptyByteArray))

  def emit(kind: WindowEvents.Kind, detail: String = ""): Unit = synchronized {
    val _ = Try(Files.write(
      target.toPath,
      (WindowEvents.line(kind, detail) + "\n").getBytes(StandardCharsets.UTF_8),
      StandardOpenOption.CREATE, StandardOpenOption.APPEND
    ))
  }
