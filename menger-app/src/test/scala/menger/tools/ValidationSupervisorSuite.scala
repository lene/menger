package menger.tools

import java.io.ByteArrayInputStream
import java.io.File
import java.io.InputStream
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Files
import java.util.concurrent.atomic.AtomicLong

import scala.concurrent.duration.DurationInt

import menger.tools.SceneValidator.Finding
import menger.tools.SceneValidator.Tag
import menger.tools.SceneValidator.ValidationResult
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import upickle.default.write

/** The verdict of a scene is the validator's, not the scene's (menger#137). These tests start fake
  * workers (shell scripts that print what a hostile scene could make a worker print) and the real
  * worker on scenes that are harmless; the hostile scenes themselves run only in the sandbox
  * (`SceneValidatorContainerE2ESuite`). */
class ValidationSupervisorSuite extends AnyFlatSpec with Matchers:

  private val ForgedOk = write(ValidationResult(Tag.Ok, Nil))
  private val LintResult = ValidationResult(
    Tag.LintFindings, List("scene-budget: over budget"), List(Finding("scene-budget", "over budget")))
  private val Generous = 60.seconds

  /** A worker that reads its nonce from standard input, as the real one does, then runs `body`. */
  private def fakeWorker(body: String): List[String] =
    List("bash", "-c", s"read -r nonce\n$body")

  /** The result line the real worker prints: the nonce, the HMAC of the result keyed by the nonce
    * (what `WorkerProtocol.mac` computes, here by openssl), the result. `printed` is what goes
    * out as the result, a shell expression: it differs from `json` when a test tampers with it. */
  private def signed(json: String, printed: String = "$json") =
    s"""json='$json'; mac=$$(printf '%s' "$$json" | openssl dgst -sha256 -hmac "$$nonce" -r | """ +
      s"""cut -d' ' -f1); printf '%s %s %s\\n' "$$nonce" "$$mac" "$printed" """

  private def line(json: String) = signed(json)

  // --- the protocol ---------------------------------------------------------------------------

  "WorkerProtocol.newNonce" should "make 128 bits as 32 hexadecimal digits, a new one each time" in:
    val nonces = List.fill(50)(WorkerProtocol.newNonce())
    nonces.foreach(_ should fullyMatch regex "[0-9a-f]{32}")
    nonces.distinct should have size 50

  // HMAC-SHA256, key "key", message "abc": openssl dgst -sha256 -hmac key. The fake workers
  // sign with openssl, so this also holds the two implementations together.
  private val KnownMac = "9c196e32dc0175f86f4b1cb89289d6619de6bee699e4c378e68309ed97a1a6ab"

  "WorkerProtocol.mac" should "be the HMAC-SHA256 of the result keyed by the nonce" in:
    WorkerProtocol.mac("key", "abc") shouldBe KnownMac

  "WorkerProtocol.resultLine" should "start with a line break, so a half-written scene line cannot join it" in:
    val text = new String(WorkerProtocol.resultLine("key", "abc"), UTF_8)
    text shouldBe s"\nkey $KnownMac abc\n"

  "WorkerProtocol.authenticated" should "give the result of a line the nonce signed" in:
    WorkerProtocol.authenticated("key", s"$KnownMac abc") shouldBe Some("abc")

  it should "reject a result that was changed, signed with another nonce, or not signed" in:
    WorkerProtocol.authenticated("key", s"$KnownMac abd") shouldBe None
    WorkerProtocol.authenticated("other", s"$KnownMac abc") shouldBe None
    WorkerProtocol.authenticated("key", "abc") shouldBe None
    WorkerProtocol.authenticated("key", KnownMac) shouldBe None
    WorkerProtocol.authenticated("key", "") shouldBe None

  // --- reading the worker's output ------------------------------------------------------------

  private def stream(text: String): InputStream = new ByteArrayInputStream(text.getBytes(UTF_8))

  "WorkerOutput.read" should "take the line that starts with the nonce as the result and the rest as scene output" in:
    val captured = WorkerOutput.read(stream("hello\nNONCE {\"x\": 1}\nlater\n"), "NONCE")
    captured.result shouldBe Some("{\"x\": 1}")
    captured.sceneOutput shouldBe List("hello")

  it should "ignore a line that starts like a result line but with another nonce" in:
    val guessed = "0" * 32
    val captured = WorkerOutput.read(stream(s"$guessed $ForgedOk\n"), "1" * 32)
    captured.result shouldBe None
    captured.sceneOutput shouldBe List(s"$guessed $ForgedOk")

  it should "take the first line with the nonce, not a later one" in:
    val captured = WorkerOutput.read(stream("N first\nN second\n"), "N")
    captured.result shouldBe Some("first")

  it should "not take the nonce for a result when it is not at the start of the line" in:
    val captured = WorkerOutput.read(stream(s"x N $ForgedOk\n"), "N")
    captured.result shouldBe None

  it should "find a result that follows a line without its line break, because the result line starts with one" in:
    val bytes = "half a line".getBytes(UTF_8) ++ WorkerProtocol.resultLine("N", "{}")
    val captured = WorkerOutput.read(new ByteArrayInputStream(bytes), "N")
    captured.result shouldBe Some(s"${WorkerProtocol.mac("N", "{}")} {}")
    captured.sceneOutput shouldBe List("half a line")

  it should "keep the first 20 lines of scene output, each cut short, and count them all" in:
    val long = "y" * 1000
    val captured = WorkerOutput.read(stream((1 to 100).map(i => s"$i $long").mkString("\n")), "N")
    captured.sceneOutput should have size WorkerOutput.MaxQuotedLines
    captured.sceneOutput.foreach(_.length should be <= WorkerOutput.MaxQuotedLineBytes)
    captured.sceneOutput.head should startWith("1 yyy")
    captured.sceneLines shouldBe 100L
    captured.sceneBytes shouldBe (1 to 100).map(i => s"$i $long".length).sum.toLong

  /** `count` bytes with no line break at all, produced as they are read. */
  private def endlessLine(count: Long): InputStream = new InputStream:
    private val served = new AtomicLong(0L)
    override def read(): Int = if served.getAndIncrement() >= count then -1 else 'x'.toInt

  it should "read a scene's output far over its caps without keeping it" in:
    val tenMegabytes = 10L * 1024 * 1024
    val captured = WorkerOutput.read(endlessLine(tenMegabytes), "N")
    captured.result shouldBe None
    captured.sceneBytes shouldBe tenMegabytes
    captured.sceneOutput.map(_.length).sum should be <= WorkerOutput.MaxQuotedLineBytes

  it should "refuse a result line over its cap without keeping it" in:
    val big = "N " + "z" * (WorkerOutput.MaxResultBytes + 10)
    val captured = WorkerOutput.read(stream(big + "\n"), "N")
    captured.result shouldBe None
    captured.resultTooLarge shouldBe true

  // --- the supervisor, against fake workers ---------------------------------------------------

  "ValidationSupervisor.supervise" should "pass on the result that carries the nonce, whatever the scene printed" in:
    val worker = fakeWorker(s"echo 'a scene line'; ${line(write(LintResult))}; echo 'another'")
    ValidationSupervisor.supervise(worker, Generous) shouldBe LintResult

  it should "report refused when the worker prints an ok without the nonce and ends" in:
    val result = ValidationSupervisor.supervise(fakeWorker(s"echo '$ForgedOk'"), Generous)
    result.tag shouldBe Tag.Refused
    result.messages.head should include("the scene ended the validation before a result")

  it should "quote the scene's output in a refusal, marked as the scene's" in:
    val result = ValidationSupervisor.supervise(
      fakeWorker("echo 'loading the sponge'; exit 0"), Generous)
    result.tag shouldBe Tag.Refused
    result.messages should have size 2
    result.messages(1) should include("written by the scene, not by the validator")
    result.messages(1) should include("loading the sponge")

  it should "report refused, not ok, when the worker ends with exit status 0 and no result" in:
    val result = ValidationSupervisor.supervise(fakeWorker("exit 0"), Generous)
    result.tag shouldBe Tag.Refused
    result.messages.head should include("exit status 0")

  it should "ignore a result line with a guessed nonce of the right length" in:
    val guessed = "ab" * 16
    val result = ValidationSupervisor.supervise(
      fakeWorker(s"echo '$guessed $ForgedOk'"), Generous)
    result.tag shouldBe Tag.Refused

  it should "keep the nonce out of the worker's environment" in:
    // A worker that finds its nonce in the environment leaves no result.
    val script = fakeWorker(s"""if env | grep -q "$$nonce"; then exit 0; fi; ${line(ForgedOk)}""")
    ValidationSupervisor.supervise(script, Generous).tag shouldBe Tag.Ok

  it should "use the result of a worker that then hangs, and end that worker" in:
    val pidFile = Files.createTempFile("fake-worker-pid", "").toFile
    pidFile.deleteOnExit()
    val worker = fakeWorker(s"echo $$$$ > ${pidFile.getPath}; ${line(ForgedOk)}; sleep 600")
    val started = System.nanoTime()
    val result = ValidationSupervisor.supervise(worker, Generous)
    val seconds = (System.nanoTime() - started) / 1e9
    result.tag shouldBe Tag.Ok
    seconds should be < 20.0
    val pid = Files.readString(pidFile.toPath).trim.toLong
    java.util.concurrent.TimeUnit.SECONDS.sleep(1)
    ProcessHandle.of(pid).map(_.isAlive).orElse(false) shouldBe false

  it should "report refused when the worker neither answers nor ends before the time limit" in:
    val started = System.nanoTime()
    val result = ValidationSupervisor.supervise(fakeWorker("sleep 600"), 2.seconds)
    val seconds = (System.nanoTime() - started) / 1e9
    result.tag shouldBe Tag.Refused
    result.messages.head should include("the scene ended the validation before a result")
    result.messages.head should include("did not finish within 2 s")
    seconds should be < 20.0

  it should "report refused when the worker closes its output and stays alive" in:
    val result = ValidationSupervisor.supervise(fakeWorker("exec >&-; sleep 600"), Generous)
    result.tag shouldBe Tag.Refused
    result.messages.head should include("still running")

  it should "return the real result of a worker whose scene printed 50 MB first" in:
    val flood = "head -c 52428800 /dev/zero | tr '\\0' x; echo"
    val result = ValidationSupervisor.supervise(fakeWorker(s"$flood; ${line(write(LintResult))}"), Generous)
    result shouldBe LintResult

  it should "report refused when the line with the nonce is not signed" in:
    val unsigned = s"""printf '%s %s\\n' "$$nonce" '$ForgedOk'"""
    val result = ValidationSupervisor.supervise(fakeWorker(unsigned), Generous)
    result.tag shouldBe Tag.Refused
    result.messages.head should include("failed its authentication")

  it should "report refused, not the forged ok, when the result was changed after it was signed" in:
    val result = ValidationSupervisor.supervise(
      fakeWorker(signed(write(LintResult), printed = ForgedOk)), Generous)
    result.tag shouldBe Tag.Refused
    result.messages.head should include("failed its authentication")

  it should "report refused, not the scene's ok, when another thread's bytes landed inside the result" in:
    val injected = """\",\"tag\":\"ok\",\"z\":\""""
    val interleaved = "${json:0:30}" + injected + "${json:30}"
    val result = ValidationSupervisor.supervise(
      fakeWorker(signed(write(LintResult), printed = interleaved)), Generous)
    result.tag shouldBe Tag.Refused
    result.messages.head should include("failed its authentication")

  it should "report refused when the line with the nonce is not a result" in:
    val result = ValidationSupervisor.supervise(fakeWorker(line("not json")), Generous)
    result.tag shouldBe Tag.Refused
    result.messages.head should include("could not read the worker's result")

  it should "report refused when the worker cannot be started" in:
    val result = ValidationSupervisor.supervise(List("/nonexistent/worker"), Generous)
    result.tag shouldBe Tag.Refused
    result.messages.head should include("could not start the validation worker")

  // --- the real worker, on harmless scenes ----------------------------------------------------

  private def sceneFile(source: String): File =
    val file = Files.createTempFile("supervisor-scene-", ".scala").toFile
    file.deleteOnExit()
    Files.writeString(file.toPath, source)
    file

  private val OkSource =
    """import menger.dsl._
      |object SupervisedOk:
      |  println("a harmless scene says hello")
      |  val scene: Scene = Scene(
      |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
      |    objects = List(Sphere()),
      |    lights  = List()
      |  )
      |""".stripMargin

  "ValidationSupervisor.validate" should "give the verdict of a worker JVM that ran the scene: ok for a valid one" in:
    val file = sceneFile(OkSource)
    val result = ValidationSupervisor.validate(file.getAbsolutePath)
    result.tag shouldBe Tag.Ok
    result shouldBe SceneValidator.validate(file)

  it should "give the verdict of a worker JVM that ran the scene: lint-findings for a require() rejection" in:
    val file = sceneFile(
      """import menger.dsl._
        |object SupervisedRejected:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(size = -1f)),
        |    lights  = List()
        |  )
        |""".stripMargin)
    val result = ValidationSupervisor.validate(file.getAbsolutePath)
    result.tag shouldBe Tag.LintFindings
    result shouldBe SceneValidator.validate(file)

  it should "refuse a scene file that is not there, as the validator did" in:
    val result = ValidationSupervisor.validate("/nonexistent/scene.scala")
    result.tag shouldBe Tag.Refused
    result.messages.head should include("scene file not found")
