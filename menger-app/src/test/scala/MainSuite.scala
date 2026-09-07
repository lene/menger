import scala.jdk.CollectionConverters._

import menger.MengerCLIOptions
import menger.engines.InteractiveEngine
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MainSuite extends AnyFlatSpec with Matchers:

  "getConfig" should "return default config if no options" in :
    val options = MengerCLIOptions(Seq.empty)
    Main.getConfig(options)

  "createEngine" should "return InteractiveEngine when --objects is set" in:
    val options = MengerCLIOptions(Seq("--objects", "type=sphere"))
    Main.createEngine(options) shouldBe a [InteractiveEngine]

  it should "return InteractiveEngine when --objects specifies a cube" in:
    val options = MengerCLIOptions(Seq("--objects", "type=cube"))
    Main.createEngine(options) shouldBe a [InteractiveEngine]

  // === --display re-exec (story 8: interactive render window) ===
  // These assert the pure ProcessBuilder-construction function only -- never Process.start()
  // -- since this environment has no real X display for a child process to render into.

  "buildReExecProcessBuilder" should "set DISPLAY in the child's environment" in:
    val builder = Main.buildReExecProcessBuilder(Array("--objects", "type=sphere"), ":1")
    builder.environment().get("DISPLAY") shouldEqual ":1"

  it should "invoke Main as the child's main class" in:
    val builder = Main.buildReExecProcessBuilder(Array.empty, ":1")
    builder.command().asScala should contain("Main")

  it should "forward the original arguments to the child" in:
    val builder = Main.buildReExecProcessBuilder(Array("--objects", "type=sphere"), ":1")
    val command = builder.command().asScala
    command should contain("--objects")
    command should contain("type=sphere")

  it should "strip a bare --display flag and its value before forwarding to the child" in:
    val builder = Main.buildReExecProcessBuilder(Array("--display", ":0", "--headless"), ":1")
    val command = builder.command().asScala
    command should not contain "--display"
    command should not contain ":0"
    command should contain("--headless")

  it should "strip a --display=value flag before forwarding to the child" in:
    val builder = Main.buildReExecProcessBuilder(Array("--display=:0", "--headless"), ":1")
    val command = builder.command().asScala
    command.exists(_.startsWith("--display")) shouldBe false
    command should contain("--headless")

  it should "leave the args unchanged when no --display flag is present" in:
    val builder = Main.buildReExecProcessBuilder(Array("--headless", "--save-name", "out.png"), ":1")
    val command = builder.command().asScala
    command should contain("--headless")
    command should contain("--save-name")
    command should contain("out.png")

  it should "strip every occurrence when --display appears more than once" in:
    val builder = Main.buildReExecProcessBuilder(
      Array("--display", ":0", "--headless", "--display=:2"), ":1"
    )
    val command = builder.command().asScala
    command should not contain "--display"
    command should not contain ":0"
    command.exists(_.startsWith("--display")) shouldBe false
    command should contain("--headless")

  it should "build a classpath via RestrictedClasspath, not the bare java.class.path property" in:
    val builder = Main.buildReExecProcessBuilder(Array.empty, ":1")
    val command = builder.command().asScala
    val cpIndex = command.indexOf("-cp")
    cpIndex should be >= 0
    command(cpIndex + 1) shouldEqual menger.dsl.RestrictedClasspath.fullClasspath

  // === shouldLock / refusedResultJson (story 8 review round: extracted for direct testability) ===

  "shouldLock" should "be true for an interactive, non-headless engine" in:
    val opts = MengerCLIOptions(Seq("--objects", "type=sphere"))
    val engine = Main.createEngine(opts)
    Main.shouldLock(engine, opts) shouldBe true

  it should "be false when --headless is set, even for an InteractiveEngine" in:
    val opts = MengerCLIOptions(Seq("--objects", "type=sphere", "--headless", "--save-name", "out.png"))
    val engine = Main.createEngine(opts)
    engine shouldBe a [InteractiveEngine]
    Main.shouldLock(engine, opts) shouldBe false

  "refusedResultJson" should "encode the reason under the wire-format 'refused' tag" in:
    val json = Main.refusedResultJson("render lock already held: /tmp/menger-render.lock")
    json should include("\"refused\"")
    json should include("render lock already held: /tmp/menger-render.lock")
