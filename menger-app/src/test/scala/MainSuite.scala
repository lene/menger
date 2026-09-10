import scala.jdk.CollectionConverters._

import menger.MengerCLIOptions
import menger.engines.InteractiveEngine
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MainSuite extends AnyFlatSpec with Matchers:

  private def freshLockPath(): String =
    java.nio.file.Files.createTempDirectory("main-suite-lock").resolve("render.lock").toString

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
    val classpath = command(cpIndex + 1)
    classpath shouldEqual menger.dsl.RestrictedClasspath.fullClasspath
    // Review round 2, partially: the assertion above compares the function under test to
    // itself, so it cannot by itself catch a revert to `System.getProperty("java.class.path")`.
    // The union invariant below is the strongest thing assertable from *here* -- `Test / fork`
    // starts this suite from a plain `java -cp`, so its classloader chain contributes nothing
    // and the two sources genuinely coincide in this JVM. A "must differ" assertion would be
    // false here, not merely weak. What actually protects the packaged/container path is
    // RestrictedClasspathSuite's stage-layout case.
    val sep = java.io.File.pathSeparator
    val systemEntries = System.getProperty("java.class.path").split(sep).filter(_.nonEmpty).toSet
    withClue("fullClasspath must be a superset of the java.class.path property: "):
      classpath.split(sep).toSet should contain allElementsOf systemEntries

  // Review round 2: the child was started as a bare `java -cp <cp> Main <args>`, dropping every
  // -D and -X the parent runs with -- including build.sbt's -Djava.library.path, without which
  // the re-exec'd child cannot load libmengergeometry.so and the window it exists to open never
  // renders.
  it should "forward the parent JVM's own options to the child" in:
    val builder = Main.buildReExecProcessBuilder(Array.empty, ":1")
    val command = builder.command().asScala.toList
    val cpIndex = command.indexOf("-cp")
    val beforeClasspath = command.slice(1, cpIndex)
    beforeClasspath shouldEqual Main.inheritedJvmOptions
    // The forwarded options must precede -cp/Main, or java treats them as program arguments.
    command.head should endWith("java")

  it should "not forward a debugger or profiler agent to the child" in:
    Main.inheritedJvmOptions.filter(o =>
      o.startsWith("-agentlib:") || o.startsWith("-agentpath:") || o.startsWith("-javaagent:")
    ) shouldBe empty

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

  // Review round 2: shouldLock, RenderLock.tryAcquire and refusedResultJson were each tested
  // in isolation and nothing composed them, so deleting Main's entire lock branch left every
  // test green. These pin the composition -- AD-16's actual behaviour.
  "acquireLockIfNeeded" should "not take a lock for a batch (headless) render" in:
    val opts = MengerCLIOptions(
      Seq("--objects", "type=sphere", "--headless", "--save-name", "out.png")
    )
    Main.acquireLockIfNeeded(Main.createEngine(opts), opts) shouldBe None

  it should "acquire the lock for an interactive render, and release it on close" in:
    val lockPath = freshLockPath()
    val opts = MengerCLIOptions(Seq("--objects", "type=sphere", "--render-lock-path", lockPath))
    val first = Main.acquireLockIfNeeded(Main.createEngine(opts), opts)
    first.map(_.isRight) shouldBe Some(true)
    first.foreach(_.foreach(_.close()))
    // Released: a second attempt on the same path now succeeds.
    val second = Main.acquireLockIfNeeded(Main.createEngine(opts), opts)
    second.map(_.isRight) shouldBe Some(true)
    second.foreach(_.foreach(_.close()))

  it should "refuse an interactive render while the lock is already held" in:
    val lockPath = freshLockPath()
    val opts = MengerCLIOptions(Seq("--objects", "type=sphere", "--render-lock-path", lockPath))
    val held = menger.engines.RenderLock.tryAcquire(lockPath)
    held shouldBe a[Right[?, ?]]
    try
      val refused = Main.acquireLockIfNeeded(Main.createEngine(opts), opts)
      refused.map(_.isLeft) shouldBe Some(true)
      refused.foreach(_.left.foreach { reason =>
        reason should include(lockPath)
        // The refusal reaches the user as AD-5's tagged JSON, not a bare message.
        Main.refusedResultJson(reason) should include("\"refused\"")
      })
    finally held.foreach(_.close())

  "refusedResultJson" should "encode the reason under the wire-format 'refused' tag" in:
    val json = Main.refusedResultJson("render lock already held: /tmp/menger-render.lock")
    json should include("\"refused\"")
    json should include("render lock already held: /tmp/menger-render.lock")
