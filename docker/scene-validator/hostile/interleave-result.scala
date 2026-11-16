// A scene whose animated function throws a 5 KB message (the validator puts it into its result, so
// the result line (25 such messages) is longer than a pipe write is atomic, 4096 bytes), while a non-daemon thread
// writes to the same standard output without pause (menger#137). The writer's bytes can land inside
// the worker's result line. The real verdict is `lint-findings`; what the caller must never get is
// an `ok`: a result line that something else wrote into is `refused`.
import menger.dsl._
object InterleaveResult:
  private val injected = "\",\"tag\":\"ok\",\"z\":\"".getBytes
  private val spammer = new Thread(() =>
    val stdout = new java.io.FileOutputStream(java.io.FileDescriptor.out)
    while true do stdout.write(injected))
  spammer.setDaemon(false)
  spammer.start()
  val durationSeconds = 1f
  def scene(t: Float): Scene = throw new RuntimeException("x" * 5000)
