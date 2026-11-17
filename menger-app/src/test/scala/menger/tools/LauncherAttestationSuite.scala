package menger.tools

import java.io.File
import java.nio.file.Files
import java.nio.file.attribute.PosixFilePermissions

import scala.jdk.CollectionConverters.*
import scala.sys.process.Process
import scala.sys.process.ProcessLogger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** `--attest` of the two launchers (menger-scene-agent#58, plan 07 Part B): the scene agent used
  * to trust a launcher by its file name. A launcher now prints one JSON line saying how it runs
  * its container, built from the very argument list it passes to `docker run`, and the agent
  * asserts that line instead.
  *
  * The tests run the real scripts against a stub `docker` that logs the arguments of `run` and
  * answers `image inspect` with a digest, so they need neither a daemon nor an image.
  */
class LauncherAttestationSuite extends AnyFlatSpec with Matchers:

  private val Digest = "sha256:" + "c" * 64

  private val dockerDir: File =
    Iterator.iterate(new File(sys.props("user.dir")).getAbsoluteFile)(_.getParentFile)
      .takeWhile(_ != null) // scalafix:ok DisableSyntax.null
      .map(dir => new File(dir, "docker/scene-validator"))
      .find(_.isDirectory)
      .getOrElse(fail("docker/scene-validator not found above the working directory"))

  private def script(name: String): File = new File(dockerDir, name)

  /** A `docker` that logs each call's arguments (one line per call, in `calls`) and the arguments
    * of `run` (one per line, in `run-args`), and answers `image inspect` with `digest` (or fails
    * when it is None, as for an image that is not there). */
  private def stubDocker(digest: Option[String] = Some(Digest)): File =
    val dir = Files.createTempDirectory("attest-stub-").toFile
    dir.deleteOnExit()
    val stub = new File(dir, "docker")
    val inspect = digest.fold("exit 1")(d => s"echo '$d'")
    Files.writeString(stub.toPath,
      s"""#!/usr/bin/env bash
         |echo "$$*" >> "${dir.getPath}/calls"
         |case "$$1" in
         |  image) $inspect ;;
         |  run) printf '%s\\n' "$$@" > "${dir.getPath}/run-args"
         |       echo '{"tag": "ok", "messages": [], "findings": [], "schemaVersion": "2.0.0"}' ;;
         |esac
         |""".stripMargin)
    Files.setPosixFilePermissions(stub.toPath, PosixFilePermissions.fromString("rwxr-xr-x"))
    dir

  private def lines(dir: File, name: String): List[String] =
    val file = new File(dir, name)
    if file.isFile then Files.readAllLines(file.toPath).asScala.toList else Nil

  private def run(
    stub: File, command: Seq[String], env: (String, String)*
  ): (Int, String) =
    val out = StringBuilder()
    val path = "PATH" -> s"${stub.getPath}:${sys.env.getOrElse("PATH", "")}"
    val exit = Process(command, None, (path +: env)*)
      .!(ProcessLogger(line => out.append(line).append('\n'), _ => ()))
    (exit, out.toString)

  private def attest(
    name: String, stub: File, extra: Seq[String] = Nil, env: Seq[(String, String)] = Nil
  ): (Int, String) =
    run(stub, Seq("bash", script(name).getPath, "--attest") ++ extra, env*)

  private def valueAfter(flag: String, args: List[String]): String =
    args.dropWhile(_ != flag).drop(1).headOption.getOrElse(fail(s"no $flag in $args"))

  "run-sandboxed.sh --attest" should "print one JSON line with the sandbox, network, root file system, image and digest" in:
    val stub = stubDocker()
    val (exit, out) = attest("run-sandboxed.sh", stub)
    exit shouldBe 0
    out.trim.linesIterator.size shouldBe 1
    val line = ujson.read(out).obj
    line.keySet shouldBe Set("sandbox", "network", "rootfs", "image", "image_digest")
    line("sandbox").str shouldBe "docker"
    line("network").str shouldBe "none"
    line("rootfs").str shouldBe "read-only"
    line("image").str shouldBe "menger-scene-validator:latest"
    line("image_digest").str shouldBe Digest

  it should "print the fields, and the required values, that contracts/process-result.json names" in:
    val contractFile = new File(dockerDir.getParentFile.getParentFile, "contracts/process-result.json")
    val spec = ujson.read(Files.readString(contractFile.toPath))("launcherAttestation")
    val line = ujson.read(attest("run-sandboxed.sh", stubDocker())._2).obj
    line.keySet shouldBe spec("requiredFields").obj.keySet
    spec("required").obj.foreach((field, value) => line(field).str shouldBe value.str)

  it should "start no container" in:
    val stub = stubDocker()
    attest("run-sandboxed.sh", stub)
    lines(stub, "calls").filter(_.startsWith("run")) shouldBe empty

  it should "attest the image it is given" in:
    val stub = stubDocker()
    val (exit, out) = attest("run-sandboxed.sh", stub, Seq("--image", "mine:test"))
    exit shouldBe 0
    ujson.read(out)("image").str shouldBe "mine:test"
    lines(stub, "calls") should contain ("image inspect --format {{.Id}} mine:test")

  it should "attest the same network and root file system as the docker run it performs" in:
    val stub = stubDocker()
    val scene = Files.createTempFile("attest-scene-", ".scala").toFile
    scene.deleteOnExit()
    val (exit, _) = run(stub, Seq("bash", script("run-sandboxed.sh").getPath, scene.getPath))
    exit shouldBe 0
    val runArgs = lines(stub, "run-args")
    val line = ujson.read(attest("run-sandboxed.sh", stub)._2).obj
    valueAfter("--network", runArgs) shouldBe line("network").str
    (runArgs should contain ("--read-only"))
    line("rootfs").str shouldBe "read-only"

  it should "refuse, and attest nothing, when the image is not there" in:
    val (exit, out) = attest("run-sandboxed.sh", stubDocker(digest = None))
    exit shouldBe 2
    val result = ujson.read(out).obj
    result("tag").str shouldBe "refused"
    result.contains("sandbox") shouldBe false

  "run-render-window.sh --attest" should "print one JSON line, without a display, for the image in MENGER_RENDER_IMAGE" in:
    val stub = stubDocker()
    val (exit, out) = attest("run-render-window.sh", stub,
      env = Seq("MENGER_RENDER_IMAGE" -> "window:test", "DISPLAY" -> ""))
    exit shouldBe 0
    out.trim.linesIterator.size shouldBe 1
    val line = ujson.read(out).obj
    line("sandbox").str shouldBe "docker"
    line("network").str shouldBe "none"
    line("rootfs").str shouldBe "read-only"
    line("image").str shouldBe "window:test"
    line("image_digest").str shouldBe Digest
    lines(stub, "calls").filter(_.startsWith("run")) shouldBe empty

  // The window's `docker run` needs a host X server, which a test cannot rely on: hold the script
  // to using the shared containment array (attestation.sh) for its flags, as the attestation is
  // built from that array.
  private val ContainmentFlags = List("--network", "--read-only", "--cap-drop", "--security-opt")

  for name <- List("run-sandboxed.sh", "run-render-window.sh") do
    s"$name" should "take its containment flags from the shared array, not spell them itself" in:
      val text = Files.readString(script(name).toPath)
      val code = text.linesIterator.filterNot(_.trim.startsWith("#")).mkString("\n")
      ContainmentFlags.foreach(flag => withClue(flag) { code should not include flag })
      code should include ("\"${CONTAINMENT[@]}\"")

  "attestation.sh" should "build the attestation from the array docker run gets" in:
    val stub = stubDocker()
    val probe =
      s""". "${script("attestation.sh").getPath}"
         |containment_flags 1g 2 3 rw,noexec
         |printf '%s\\n' "$${CONTAINMENT[@]}"
         |echo ---
         |attestation some:image "$${CONTAINMENT[@]}"
         |""".stripMargin
    val (exit, out) = run(stub, Seq("bash", "-c", probe))
    exit shouldBe 0
    val (flags, attested) = out.linesIterator.toList.span(_ != "---")
    val line = ujson.read(attested.drop(1).mkString("\n")).obj
    valueAfter("--network", flags) shouldBe line("network").str
    flags should contain ("--read-only")
    line("rootfs").str shouldBe "read-only"

  it should "report a root file system that docker run would leave writable as writable" in:
    val stub = stubDocker()
    val probe =
      s""". "${script("attestation.sh").getPath}"
         |attestation some:image --network none --memory 1g
         |""".stripMargin
    val (_, out) = run(stub, Seq("bash", "-c", probe))
    ujson.read(out)("rootfs").str shouldBe "writable"

  it should "report a network that docker run would leave on as the network docker run gets" in:
    val stub = stubDocker()
    val probe =
      s""". "${script("attestation.sh").getPath}"
         |attestation some:image --read-only --network bridge
         |""".stripMargin
    val (_, out) = run(stub, Seq("bash", "-c", probe))
    ujson.read(out)("network").str shouldBe "bridge"

  it should "report no --network flag as docker's default, not as none" in:
    val stub = stubDocker()
    val probe =
      s""". "${script("attestation.sh").getPath}"
         |attestation some:image --read-only
         |""".stripMargin
    val (_, out) = run(stub, Seq("bash", "-c", probe))
    ujson.read(out)("network").str shouldBe "default"
