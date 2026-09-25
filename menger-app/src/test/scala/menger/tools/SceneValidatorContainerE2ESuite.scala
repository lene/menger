package menger.tools

import java.io.File
import java.io.PrintWriter
import java.nio.file.Files

import scala.concurrent.Await
import scala.concurrent.ExecutionContext
import scala.concurrent.Future
import scala.concurrent.TimeoutException
import scala.concurrent.duration.Duration
import scala.concurrent.duration.SECONDS
import scala.sys.process.Process
import scala.sys.process.ProcessLogger
import scala.util.Try

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Real end-to-end test of the AD-18 sandbox -- the story's own acceptance criterion ("one
  * real end-to-end test actually running the container against a known-good scene and a
  * known-bad one," `spec-ai-scene-agent/stories/5-gauntlet-renderer-side.md`'s Tasks &
  * Acceptance) as an automated test, not a manual step performed once. Actually shells out to
  * `docker/scene-validator/run-sandboxed.sh`, which runs the real built
  * `menger-scene-validator:latest` image with `--network none`, `--cap-drop ALL`,
  * `--read-only`, resource/wall-clock bounds and a single read-only scene-file bind mount,
  * against a genuine temp scene file on disk.
  *
  * Skips via `assume` (not a failure) when `docker` isn't on PATH, the run script can't be
  * found, or the image hasn't been built (`./docker/scene-validator/build.sh`) -- the story's
  * own `Never` clause keeps this image out of CI, so a machine without it is an expected
  * environment, not an exceptional one. Mirrors `VideoEncoderSuite`'s `assume`-gated pattern
  * for ffmpeg/libx264 availability -- the one other place this codebase gates a test on an
  * external tool that may or may not be present on the running machine.
  */
class SceneValidatorContainerE2ESuite extends AnyFlatSpec with Matchers:

  private val ImageTag = "menger-scene-validator:latest"

  /** Upper bound on one sandboxed run. `run-sandboxed.sh` enforces its own 120s wall clock and
    * now tears the container down itself, but a test must never depend on the thing it is
    * testing to terminate: an unbounded `Process(...).!` here blocks the suite forever if the
    * wrapper's teardown regresses (review round 2). */
  private val RunTimeoutSeconds = 300L

  private def dockerAvailable: Boolean =
    Try(Process(Seq("docker", "--version")).!(ProcessLogger(_ => ()))).toOption.contains(0)

  private def imageBuilt: Boolean =
    Try(Process(Seq("docker", "image", "inspect", ImageTag)).!(ProcessLogger(_ => ()))).toOption.contains(0)

  private def ancestors(dir: File): LazyList[File] =
    dir #:: Option(dir.getParentFile).map(ancestors).getOrElse(LazyList.empty)

  /** `run-sandboxed.sh` lives at `<repo-root>/docker/scene-validator/`; sbt's `Test` task
    * runs with `mengerApp/`'s own base directory as the JVM's working directory (unlike
    * `run`, whose `baseDirectory` is overridden to the repo root in `build.sbt`), so this
    * walks up from the working directory rather than assuming either layout. */
  private def findRunScript(): Option[File] =
    ancestors(new File(".").getAbsoluteFile)
      .take(6)
      .map(dir => new File(dir, "docker/scene-validator/run-sandboxed.sh"))
      .find(_.isFile)

  /** The mounted scene directory is read inside the container by `scene-validator`, a uid
    * distinct from the host user running this test (see `Dockerfile`'s `useradd`) -- Docker's
    * bind mount preserves host file modes as-is, so `Files.createTempDirectory`'s default
    * `0700` would make the mount unreadable to that uid and every scene file resolve as
    * "not found" (a `Refused`, not the tag under test). Both the directory and the file are
    * widened to world-readable/traversable; this is disposable scratch content, never
    * sensitive, so the loosened mode carries no real risk. */
  private def writeTempScene(content: String): File =
    val dir = Files.createTempDirectory("scene-validator-e2e-").toFile
    dir.deleteOnExit()
    dir.setReadable(true, false)
    dir.setExecutable(true, false)
    val f = new File(dir, "scene.scala")
    f.deleteOnExit()
    val pw = new PrintWriter(f)
    try pw.write(content)
    finally pw.close()
    f.setReadable(true, false)
    f

  private def runSandboxed(script: File, sceneFile: File): (Int, String) =
    val out = new StringBuilder
    val logger = ProcessLogger(
      line => { out.append(line); out.append("\n"); () },
      line => { out.append(line); out.append("\n"); () }
    )
    val running = Process(Seq(script.getAbsolutePath, sceneFile.getAbsolutePath)).run(logger)
    val finished = Future(running.exitValue())(ExecutionContext.global)
    val exitCode =
      try Await.result(finished, Duration(RunTimeoutSeconds, SECONDS))
      catch
        case _: TimeoutException =>
          running.destroy()
          fail(
            s"run-sandboxed.sh did not finish within ${RunTimeoutSeconds}s -- its own wall-clock " +
              s"bound should have fired long before. Output so far:\n${out.toString}"
          )
    (exitCode, out.toString)

  private def assumeSandboxAvailable(script: Option[File]): Unit =
    assume(dockerAvailable, "docker not available on this system -- skipping container e2e test")
    assume(script.isDefined, "docker/scene-validator/run-sandboxed.sh not found -- skipping")
    assume(imageBuilt, s"$ImageTag not built (run docker/scene-validator/build.sh first) -- skipping")

  "the real AD-18 sandbox" should "return ok for a known-good scene run through the real container" in:
    val script = findRunScript()
    assumeSandboxAvailable(script)

    val scene = writeTempScene(
      """import menger.dsl._
        |object E2EGoodScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 2f, 5f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Tesseract(Material.Glass)),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val (exitCode, output) = runSandboxed(script.get, scene)
    withClue(output) { exitCode shouldBe 0 }
    output should include("\"tag\": \"ok\"")

  it should "return CompileErrors, not ok, for a known-bad (syntax error) scene run through the real container" in:
    val script = findRunScript()
    assumeSandboxAvailable(script)

    val scene = writeTempScene("object E2EBrokenScene { THIS IS NOT SCALA !!!!")
    val (exitCode, output) = runSandboxed(script.get, scene)
    withClue(output) { exitCode should not be 0 }
    output should include("\"tag\": \"compile-errors\"")

  it should "return LintFindings, not ok, for a known-bad (require() rejection) scene run through the real container" in:
    val script = findRunScript()
    assumeSandboxAvailable(script)

    val scene = writeTempScene(
      """import menger.dsl._
        |object E2ERequireViolationScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(size = -1f)),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val (exitCode, output) = runSandboxed(script.get, scene)
    withClue(output) { exitCode should not be 0 }
    output should include("\"tag\": \"lint-findings\"")
