package menger.tools

import java.io.File
import java.lang.ProcessBuilder.Redirect
import java.lang.management.ManagementFactory
import java.nio.charset.StandardCharsets.UTF_8
import java.util.concurrent.TimeUnit

import scala.concurrent.Await
import scala.concurrent.ExecutionContext
import scala.concurrent.Future
import scala.concurrent.TimeoutException
import scala.concurrent.blocking
import scala.concurrent.duration.DurationInt
import scala.concurrent.duration.FiniteDuration
import scala.jdk.CollectionConverters._
import scala.util.Failure
import scala.util.Success
import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import menger.tools.SceneValidator.Tag
import menger.tools.SceneValidator.ValidationResult
import upickle.default.read

/** The validator process that never runs scene code (menger#137). It starts a worker JVM, which
  * compiles, loads and checks the scene ([[SceneValidatorWorker]]), and it alone prints the
  * result. A scene shares its JVM's standard output and can end that JVM, so the verdict could
  * be forged or lost in the same process; here the verdict is the worker's line that carries the
  * nonce ([[WorkerProtocol]]) or, when there is none, `refused`. It is never `ok` on the
  * supervisor's own say-so.
  *
  * What this does not stop: scene code runs with the worker's privileges, so a scene that reads
  * the nonce out of the worker's own memory (a heap dump through the management interface, for
  * one) can still print a forged line. The container is the security boundary, as before; `ok`
  * holds against a scene that is wrong, greedy or casually hostile, not against every adversary. */
object ValidationSupervisor extends LazyLogging:

  /** Below the 120 s the sandbox wrapper kills the container after: the supervisor can still
    * report `refused`, where the wrapper's kill leaves no result at all. */
  val DefaultTimeLimit: FiniteDuration = 100.seconds

  /** How long a worker that closed its output without a result gets to end by itself. */
  private val ExitGracePeriodSeconds = 2L

  private val WorkerMainClass = SceneValidatorWorker.getClass.getName.stripSuffix("$")

  private val SizingOptionPrefixes = List("-Xmx", "-Xms")

  /** The result for `scenePath`, from a worker JVM started for it. */
  def validate(scenePath: String): ValidationResult =
    supervise(workerCommand(scenePath), DefaultTimeLimit)

  /** The worker runs on the supervisor's own classpath and with its JVM options except the heap
    * size, which is the supervisor's small one (the sandbox's image sizes the worker with
    * `-XX:MaxRAMPercentage`). */
  private[tools] def workerCommand(scenePath: String): List[String] =
    val javaExecutable = new File(new File(System.getProperty("java.home"), "bin"), "java").getPath
    val inherited = ManagementFactory.getRuntimeMXBean.getInputArguments.asScala.toList
      .filterNot(option => SizingOptionPrefixes.exists(option.startsWith))
    List(javaExecutable) ++ inherited ++
      List("-cp", System.getProperty("java.class.path"), WorkerMainClass, scenePath)

  /** Runs `command` as the worker: gives it a fresh nonce on its standard input, reads its
    * standard output for the result line, and ends it as soon as that arrives (a scene may keep
    * the JVM alive with a thread) or when `timeLimit` passes. */
  private[tools] def supervise(command: List[String], timeLimit: FiniteDuration): ValidationResult =
    Try(new ProcessBuilder(command.asJava).redirectError(Redirect.INHERIT).start()) match
      case Failure(e) =>
        logger.error(s"Starting the validation worker failed: ${command.mkString(" ")}", e)
        ValidationResult(Tag.Refused, List(s"could not start the validation worker: ${e.getMessage}"))
      case Success(worker) => superviseStarted(worker, timeLimit)

  private def superviseStarted(worker: Process, timeLimit: FiniteDuration): ValidationResult =
    val nonce = WorkerProtocol.newNonce()
    sendNonce(worker, nonce)
    // A daemon thread of the global pool: if a scene's subprocess keeps the pipe open past
    // the worker's end, the read stays blocked without holding the validator up.
    val output = Future(blocking(WorkerOutput.read(worker.getInputStream, nonce)))(using
      ExecutionContext.global)
    val captured = Try(Await.result(output, timeLimit))
    val exitStatus = captured.toOption.filter(_.result.isEmpty).flatMap(_ => exitStatusIfEnded(worker))
    endProcessTree(worker)
    captured match
      case Success(c) => resultOf(c, exitStatus)
      case Failure(_: TimeoutException) =>
        refusedWithoutResult(
          s"the validation did not finish within ${timeLimit.toSeconds} s and was ended", None)
      case Failure(e) =>
        logger.error("Reading the validation worker's output failed", e)
        refusedWithoutResult(s"reading the validation worker's output failed: ${e.getMessage}", None)

  /** A worker that closed its output without a result usually ended with it; one that did not is
    * a scene holding the JVM open, and gets no more than the grace period. */
  private def exitStatusIfEnded(worker: Process): Option[Int] =
    Option.when(worker.waitFor(ExitGracePeriodSeconds, TimeUnit.SECONDS))(worker.exitValue())

  /** The worker reads the nonce before it touches the scene and never again. A worker that is
    * already gone makes this fail; that is the same as any worker with no result. */
  private def sendNonce(worker: Process, nonce: String): Unit =
    Try(worker.getOutputStream).foreach { stdin =>
      Try { stdin.write(s"$nonce\n".getBytes(UTF_8)); stdin.flush() }
      Try(stdin.close())
    }

  private def endProcessTree(worker: Process): Unit =
    worker.descendants().forEach(child => { child.destroyForcibly(); () })
    worker.destroyForcibly()
    val _ = worker.waitFor(ExitGracePeriodSeconds, TimeUnit.SECONDS)

  private def resultOf(captured: WorkerOutput.Captured, exitStatus: Option[Int]): ValidationResult =
    captured.result match
      case Some(json) =>
        Try(read[ValidationResult](json)).fold(
          e => refusedWithResultProblem(s"could not read the worker's result: ${e.getMessage}", captured),
          identity
        )
      case None if captured.resultTooLarge =>
        refusedWithResultProblem(
          s"the worker's result is longer than ${WorkerOutput.MaxResultBytes} bytes", captured)
      case None =>
        val how = exitStatus.fold("the worker was still running when it closed its output and was ended")(
          status => s"the worker ended with exit status $status")
        refusedWithoutResult(how, Some(captured))

  /** The scene ended the validation: it left no result line. */
  private def refusedWithoutResult(detail: String, captured: Option[WorkerOutput.Captured]): ValidationResult =
    ValidationResult(
      Tag.Refused,
      s"the scene ended the validation before a result ($detail)" ::
        captured.toList.flatMap(sceneOutputMessage)
    )

  private def refusedWithResultProblem(problem: String, captured: WorkerOutput.Captured): ValidationResult =
    ValidationResult(Tag.Refused, s"validation failed: $problem" :: sceneOutputMessage(captured).toList)

  /** Marked as the scene's, so that a reader (or the agent's model) does not take it for the
    * validator's own words. */
  private def sceneOutputMessage(captured: WorkerOutput.Captured): Option[String] =
    Option.when(captured.sceneOutput.nonEmpty) {
      val shown = captured.sceneOutput.size
      val header = "scene output (written by the scene, not by the validator; " +
        s"first $shown of ${captured.sceneLines} lines, ${captured.sceneBytes} bytes in all):"
      (header :: captured.sceneOutput.map("  " + _)).mkString("\n")
    }
