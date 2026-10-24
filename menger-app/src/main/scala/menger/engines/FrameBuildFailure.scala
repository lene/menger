package menger.engines

/** A frame whose scene could not be built is logged (stderr) with a fixed marker, so a caller
  * watching the log -- the scene agent's render window -- can report it; the window itself
  * keeps running and shows the previous frame (usability review 2026-09, session 2,
  * menger#54). */
object FrameBuildFailure:
  val Marker = "FRAME-BUILD-FAILED"

  def message(context: String, cause: Throwable): String =
    s"$Marker $context: ${Option(cause.getMessage).getOrElse(cause.getClass.getSimpleName)}"
