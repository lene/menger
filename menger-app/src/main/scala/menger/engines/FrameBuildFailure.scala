package menger.engines

/** A frame whose scene could not be built is logged (stderr) with a fixed marker, so a caller
  * watching the log -- the scene agent's render window -- can report it; the window itself
  * keeps running and shows the previous frame (usability review 2026-09, session 2,
  * menger#54). */
object FrameBuildFailure:
  val Marker = "FRAME-BUILD-FAILED"

  /** `FRAME-BUILD-FAILED [kind] context: message`; the kind (scene, environment, internal) is
    * [[SceneFailure]]'s classification of the cause. */
  def message(context: String, cause: Throwable): String =
    val failure = SceneFailure.of(cause)
    s"$Marker [${failure.kind}] $context: ${failure.message}"
