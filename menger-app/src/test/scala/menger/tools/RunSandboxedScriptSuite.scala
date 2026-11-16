package menger.tools

import java.io.File
import java.nio.file.Files
import java.nio.file.attribute.PosixFilePermissions

import scala.jdk.CollectionConverters.*
import scala.sys.process.Process
import scala.sys.process.ProcessLogger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** `run-sandboxed.sh --expect-dsl-revision` against a stub `docker` (usability review 2026-10,
  * session 4, F99): a stale validator image passed for current because its only label was the
  * app version. The label is the manifest's contentRevision since menger#83 (it was the whole
  * file's sha256). No real image or daemon needed, unlike `SceneValidatorContainerE2ESuite`.
  */
class RunSandboxedScriptSuite extends AnyFlatSpec with Matchers:

  private val ImageHash = "a" * 64

  private def ancestors(dir: File): LazyList[File] =
    dir #:: Option(dir.getParentFile).map(ancestors).getOrElse(LazyList.empty)

  private val script: File =
    ancestors(new File(".").getAbsoluteFile).take(6)
      .map(dir => new File(dir, "docker/scene-validator/run-sandboxed.sh"))
      .find(_.isFile)
      .getOrElse(fail("docker/scene-validator/run-sandboxed.sh not found"))

  /** A `docker` that reports `label` for `inspect`, logs every call, and answers `run` with
    * the validator's `ok` JSON. */
  private val OkJson = """{"tag": "ok", "messages": [], "findings": [], "schemaVersion": "1.0.0"}"""
  private val LintJson =
    """{"tag": "lint-findings", "messages": [], "findings": [], "schemaVersion": "1.0.0"}"""

  /** `runOutput` is what the container prints (lines) and `runExit` its exit status. */
  private def stubDocker(
    label: String, runOutput: List[String] = List(OkJson), runExit: Int = 0
  ): File =
    val dir = Files.createTempDirectory("run-sandboxed-stub-").toFile
    dir.deleteOnExit()
    val stub = new File(dir, "docker")
    val printed = if runOutput.isEmpty then ":" else runOutput.map(line => s"echo '$line'").mkString("; ")
    Files.writeString(stub.toPath,
      s"""#!/usr/bin/env bash
         |echo "$$1" >> "${dir.getPath}/calls"
         |case "$$1" in
         |  inspect) echo "$label" ;;
         |  run) $printed; exit $runExit ;;
         |esac
         |""".stripMargin)
    Files.setPosixFilePermissions(stub.toPath, PosixFilePermissions.fromString("rwxr-xr-x"))
    dir

  private def run(stubDir: File, expected: String): (Int, String, List[String]) =
    val scene = Files.createTempFile("run-sandboxed-scene-", ".scala").toFile
    scene.deleteOnExit()
    val out = StringBuilder()
    val env = "PATH" -> s"${stubDir.getPath}:${sys.env.getOrElse("PATH", "")}"
    val exit = Process(
      Seq("bash", script.getPath, scene.getPath, "--expect-dsl-revision", expected),
      None, env
    ).!(ProcessLogger(line => out.append(line).append('\n'), _ => ()))
    val calls = new File(stubDir, "calls")
    val logged = if calls.isFile then Files.readAllLines(calls.toPath).asScala.toList else Nil
    (exit, out.toString, logged)

  "run-sandboxed.sh" should "refuse without running the container when the DSL revision differs" in:
    val (exit, out, calls) = run(stubDocker(ImageHash), "b" * 64)
    exit shouldBe 2
    out should include ("\"tag\": \"refused\"")
    out should include ("different DSL manifest")
    calls should not contain "run"

  it should "refuse an image without the label" in:
    val (exit, out, calls) = run(stubDocker(""), ImageHash)
    exit shouldBe 2
    out should include ("<no label>")
    calls should not contain "run"

  it should "run the container when the hashes match" in:
    val (exit, out, calls) = run(stubDocker(ImageHash), ImageHash)
    exit shouldBe 0
    out should include ("\"tag\": \"ok\"")
    calls should contain ("run")

  // menger#137: what a scene writes to the container's stdout by another route must not pass for
  // the verdict.
  it should "print the validator's document when its tag agrees with the exit status" in:
    val (exit, out, _) = run(stubDocker(ImageHash, List(LintJson), runExit = 1), ImageHash)
    exit shouldBe 1
    out should include ("\"tag\": \"lint-findings\"")

  it should "refuse, and print nothing of what the container wrote, when it wrote two documents" in:
    val (exit, out, _) = run(stubDocker(ImageHash, List(OkJson, LintJson), runExit = 1), ImageHash)
    exit shouldBe 2
    ujson.read(out)("tag").str shouldBe "refused"
    out should not include "lint-findings"

  it should "refuse a document whose tag the exit status contradicts" in:
    val (exit, out, _) = run(stubDocker(ImageHash, List(OkJson), runExit = 1), ImageHash)
    exit shouldBe 2
    ujson.read(out)("tag").str shouldBe "refused"

  it should "refuse output that is not a document" in:
    val (exit, out, _) = run(stubDocker(ImageHash, List("this is not json"), runExit = 0), ImageHash)
    exit shouldBe 2
    ujson.read(out)("tag").str shouldBe "refused"

  it should "pass a run killed from outside through, as it has no result to check" in:
    val (exit, out, _) = run(stubDocker(ImageHash, Nil, runExit = 124), ImageHash)
    exit shouldBe 124
    out.trim shouldBe empty
