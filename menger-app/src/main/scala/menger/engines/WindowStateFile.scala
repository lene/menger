package menger.engines

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.StandardCopyOption

import scala.util.Try

import upickle.default.ReadWriter
import upickle.default.write

/** What the render window currently shows, for the scene agent to read (usability review
  * 2026-10, session 4, F96: "/ask what are the 4D rot/projection parameters now?" could only
  * answer from the file, and "keep them like that" had nothing to keep). Written next to the
  * scene file as `<scene file>.window-state.json` whenever it changes, at most every
  * `MinIntervalMillis`. Vectors are `[x, y, z]`, angles degrees. */
object WindowStateFile:

  final case class State(
    cameraEye: List[Float],
    cameraLookAt: List[Float],
    cameraUp: List[Float],
    rotXW: Float,
    rotYW: Float,
    rotZW: Float,
    t: Option[Float],
    playing: Option[Boolean]
  ) derives ReadWriter

  val MinIntervalMillis = 250L

  def pathFor(sceneFile: File): File =
    new File(sceneFile.getPath + ".window-state.json")

  def of(
    eye: menger.common.Vector[3], lookAt: menger.common.Vector[3], up: menger.common.Vector[3],
    offset: Interactive4DOffset, t: Option[Float], playing: Option[Boolean]
  ): State =
    import menger.common.x
    import menger.common.y
    import menger.common.z
    State(
      List(eye.x, eye.y, eye.z), List(lookAt.x, lookAt.y, lookAt.z), List(up.x, up.y, up.z),
      offset.rotXW, offset.rotYW, offset.rotZW, t, playing
    )

/** Writes a `State` when it differs from the last one written and `MinIntervalMillis` have
  * passed. Never throws: a failed write only loses that update. */
final class WindowStateFile(
  target: File,
  now: () => Long = () => System.currentTimeMillis()
):
  import WindowStateFile.*

  // (state written last, time of that write)
  private val lastWrite = new java.util.concurrent.atomic.AtomicReference[(Option[State], Long)](
    (None, Long.MinValue)
  )

  def update(state: State): Unit =
    val time = now()
    val (last, at) = lastWrite.get()
    if !last.contains(state) && (at == Long.MinValue || time - at >= MinIntervalMillis) then
      val written = Try:
        val tmp = new File(target.getPath + ".tmp")
        Files.write(tmp.toPath, write(state, indent = 2).getBytes(StandardCharsets.UTF_8))
        Files.move(tmp.toPath, target.toPath, StandardCopyOption.REPLACE_EXISTING)
      if written.isSuccess then lastWrite.set((Some(state), time))
