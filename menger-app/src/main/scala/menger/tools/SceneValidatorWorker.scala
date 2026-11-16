package menger.tools

import java.io.BufferedReader
import java.io.FileDescriptor
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets.UTF_8

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import menger.tools.SceneValidator.Tag
import menger.tools.SceneValidator.ValidationResult
import upickle.default.write

/** The JVM that runs a scene's code for [[ValidationSupervisor]] (menger#137): compiles, loads and
  * checks it as [[SceneValidator.run]] does, and sends the result to the supervisor on a line that
  * starts with the nonce the supervisor wrote to its standard input (see [[WorkerProtocol]]).
  *
  * The nonce is read, and the standard output captured, before the scene's code is touched, and
  * the nonce lives only in this method's frame. It is not an argument or an environment variable.
  *
  * Not for anyone's own use: the supervisor starts it. */
object SceneValidatorWorker extends LazyLogging:

  def main(args: Array[String]): Unit =
    SceneValidatorWorkerMain.readNonce() match
      case None =>
        logger.error("SceneValidatorWorker: no nonce on standard input; the supervisor starts this")
        SceneValidatorWorkerMain.endWith(WorkerMisuseExitCode)
      case Some(nonce) =>
        val resultStream = SceneValidatorWorkerMain.captureStandardOutput()
        val result = SceneValidator.run(args).fold(
          message => ValidationResult(Tag.Refused, List(message)), identity)
        val sent = Try(
          SceneValidatorWorkerMain.send(resultStream, WorkerProtocol.resultLine(nonce, write(result))))
        sent.failed.foreach(e => logger.error("SceneValidatorWorker: the result was not sent", e))
        SceneValidatorWorkerMain.endWith(if sent.isSuccess then 0 else SendFailedExitCode)

  private val WorkerMisuseExitCode = 2
  private val SendFailedExitCode = 3

/** The process-level side effects `ArchitectureSpec` restricts to classes named `.*Main.*`
  * (standard streams, ending the JVM), like `SceneValidatorMain`. */
private object SceneValidatorWorkerMain:

  /** The first line of standard input, which is then closed. */
  def readNonce(): Option[String] =
    val reader = new BufferedReader(new InputStreamReader(System.in, UTF_8))
    try Option(reader.readLine()).map(_.trim).filter(_.nonEmpty)
    finally reader.close()

  /** The process's own standard output, which the scene's `System.out` (replaceable by the scene)
    * is not: the result goes out on this one. */
  def captureStandardOutput(): FileOutputStream = new FileOutputStream(FileDescriptor.out)

  /** The scene's output so far goes out first, then the result in one write. A closed output
    * (the scene closed it) loses the result, which makes the verdict `refused`, never `ok`. */
  def send(stream: FileOutputStream, line: Array[Byte]): Unit =
    System.out.flush()
    stream.write(line)
    stream.flush()

  /** `halt`, not `exit`: the scene may have registered shutdown hooks or left threads, and
    * nothing is left to do. */
  def endWith(status: Int): Unit = Runtime.getRuntime.halt(status)
