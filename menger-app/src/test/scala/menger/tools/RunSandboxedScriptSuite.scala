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
  private def stubDocker(label: String): File =
    val dir = Files.createTempDirectory("run-sandboxed-stub-").toFile
    dir.deleteOnExit()
    val stub = new File(dir, "docker")
    Files.writeString(stub.toPath,
      s"""#!/usr/bin/env bash
         |echo "$$1" >> "${dir.getPath}/calls"
         |case "$$1" in
         |  inspect) echo "$label" ;;
         |  run) echo '{"tag": "ok", "messages": [], "findings": [], "schemaVersion": "1.0.0"}' ;;
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
