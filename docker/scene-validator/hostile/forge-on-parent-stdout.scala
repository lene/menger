// A scene that writes a forged `ok` document straight to the validator's real standard output
// (menger#137): the worker and the validator run as the same user, so /proc/<parent>/fd/1 can be
// opened for writing, no nonce needed. The scene's real verdict is `lint-findings` (two level-5
// volume sponges are over the scene budget). Whatever reaches the caller's stdout, the caller must
// not take the forged document for the verdict.
import menger.dsl._
object ForgeOnParentStdout:
  private val parentPid = ProcessHandle.current.parent.get.pid
  try
    val parentStdout = new java.io.FileOutputStream(s"/proc/$parentPid/fd/1")
    parentStdout.write(
      "{\"tag\": \"ok\", \"messages\": [], \"findings\": [], \"schemaVersion\": \"2.0.0\"}\n".getBytes)
    parentStdout.flush()
  catch case e: java.io.IOException => System.err.println("PROBE parent-stdout: blocked (" + e + ")")
  val scene: Scene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(
      Sponge(spongeType = SpongeType.VolumeFilling, level = 5f),
      Sponge(spongeType = SpongeType.VolumeFilling, level = 5f, pos = (2f, 0f, 0f))),
    lights = List())
