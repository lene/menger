package menger.scene

import java.nio.file.Files

import menger.dsl.LoadedScene
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneCompilerSuite extends AnyFlatSpec with Matchers:

  private def writeTempScene(content: String): java.io.File =
    val f  = Files.createTempFile("menger-test-", ".scala").toFile
    f.deleteOnExit()
    val pw = java.io.PrintWriter(f)
    pw.write(content)
    pw.close()
    f

  "SceneCompiler" should "compile a valid static scene file" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object StaticTestScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere()),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    SceneCompiler.compile(file) shouldBe a[Right[?, ?]]

  // menger#95: the 4D angles are Degrees, but a scene that passes plain Floats (as every
  // scene written before the types did) must keep compiling, without opting into implicit
  // conversions -- the conversion is a warning at the use site, not an error.
  it should "compile a scene that passes plain Floats for the 4D rotation angles" in:
    val file = writeTempScene(
      """import menger.dsl._
        |import menger.spec.Projection4DSpec
        |object FloatAnglesTestScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Tesseract(
        |      projection = Some(Projection4DSpec(rotXW = 30f, rotYW = 45f, rotZW = 0f))
        |    )),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    SceneCompiler.compile(file) shouldBe a[Right[?, ?]]

  it should "return Left for a file with a syntax error" in:
    val file = writeTempScene("object Broken { THIS IS NOT SCALA !!!!")
    SceneCompiler.compile(file) shouldBe a[Left[?, ?]]

  // RestrictedClasspath's own doc comment (menger.scene.RestrictedClasspath): omitting
  // `scala-logging_3` from its allowlist doesn't produce a normal compile error for a scene
  // referencing ParametricSurface -- it crashes the Scala 3 compiler with an opaque
  // `AssertionError: class X has non-class parent` during base-class linearization, because
  // ParametricSurface extends LazyLogging and the compiler must fully resolve that supertype
  // to typecheck any reference to it. That was diagnosed once, by hand, against the full
  // example corpus -- this pins it as a repeatable regression test: a future edit that
  // narrows RestrictedClasspath's allowlist (e.g. dropping scala-logging as "unused by any
  // scene file directly") would silently reintroduce exactly this failure mode for any real
  // ParametricSurface-based scene, and this test would catch it immediately.
  it should "compile a scene using ParametricSurface through the real restricted classpath" in:
    val file = writeTempScene(
      """import scala.math._
        |import menger.dsl._
        |object ParametricSurfaceTestScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(ParametricSurface(
        |      f = (u, v) => Vec3(cos(u).toFloat, sin(u).toFloat, v)
        |    )),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    SceneCompiler.compile(file) shouldBe a[Right[?, ?]]

  "SceneLoader" should "load a static scene from a .scala file path" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object RuntimeStaticScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere()),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneLoader.load(file.getAbsolutePath)
    result shouldBe a[Right[?, ?]]
    result.map {
      case LoadedScene.Static(_)   => succeed
      case LoadedScene.Animated(_) => fail("Expected Static")
    }

  it should "load an animated scene from a .scala file path" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object RuntimeAnimatedScene:
        |  def scene(t: Float): Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(pos = Vec3(t, 0f, 0f))),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneLoader.load(file.getAbsolutePath)
    result shouldBe a[Right[?, ?]]
    result.map {
      case LoadedScene.Animated(fn) =>
        val s = fn(1.0f)
        s.objects should have length 1
      case LoadedScene.Static(_) => fail("Expected Animated")
    }

  it should "return Left for a .scala file with compilation errors" in:
    val file = writeTempScene("object Broken { THIS IS NOT VALID SCALA !!!!")
    SceneLoader.load(file.getAbsolutePath) shouldBe a[Left[?, ?]]

  it should "return Left when the .scala file does not exist" in:
    SceneLoader.load("/nonexistent/path/missing.scala") shouldBe a[Left[?, ?]]

  it should "return Left when object has no scene member" in:
    val file = writeTempScene(
      """object NoSceneMember:
        |  val notAScene = 42
        |""".stripMargin
    )
    SceneLoader.load(file.getAbsolutePath) shouldBe a[Left[?, ?]]

  private def animatedScene(name: String, extraMember: String, sphereSize: String): String =
    s"""import menger.dsl._
       |object $name:
       |  $extraMember
       |  def scene(t: Float): Scene = Scene(
       |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
       |    objects = List(Sphere(size = $sphereSize)),
       |    lights  = List()
       |  )
       |""".stripMargin

  it should "read an animated scene's declared durationSeconds" in:
    val file = writeTempScene(animatedScene("TimedScene", "val durationSeconds = 10f", "1f + t"))
    SceneLoader.load(file.getAbsolutePath) match
      case Right(animated: LoadedScene.Animated) => animated.duration shouldBe Some(10f)
      case other => fail(s"Expected Animated, got $other")

  // Usability review 2026-10, session 3 (F84): `duration` did not say it is seconds; the old
  // name stays accepted for a deprecation period.
  it should "still accept the deprecated `val duration`" in:
    val file = writeTempScene(animatedScene("OldTimedScene", "val duration = 10f", "1f + t"))
    SceneLoader.load(file.getAbsolutePath) match
      case Right(animated: LoadedScene.Animated) => animated.duration shouldBe Some(10f)
      case other => fail(s"Expected Animated, got $other")

  it should "prefer durationSeconds when a scene declares both" in:
    val members = "val durationSeconds = 4f\n  val duration = 10f"
    val file = writeTempScene(animatedScene("BothTimedScene", members, "1f + t"))
    SceneLoader.load(file.getAbsolutePath) match
      case Right(animated: LoadedScene.Animated) => animated.duration shouldBe Some(4f)
      case other => fail(s"Expected Animated, got $other")

  it should "leave the duration empty when an animated scene declares none" in:
    val file = writeTempScene(animatedScene("UntimedScene", "", "1f + t"))
    SceneLoader.load(file.getAbsolutePath) match
      case Right(animated: LoadedScene.Animated) => animated.duration shouldBe None
      case other => fail(s"Expected Animated, got $other")

  it should "reject a non-positive duration" in:
    val file =
      writeTempScene(animatedScene("NegativeDurationScene", "val durationSeconds = -1f", "1f + t"))
    SceneLoader.load(file.getAbsolutePath) shouldBe Left("'durationSeconds' must be positive, got -1.0")

  // Usability review 2026-09: the loader swallowed exceptions from probing scene(0), so a scene
  // whose require() failed at t=0 was reported as having no scene method at all.
  it should "report the scene's own error when scene(0) throws" in:
    val file = writeTempScene(animatedScene("ThrowingAtZeroScene", "", "t"))
    val result = SceneLoader.load(file.getAbsolutePath)
    result.left.map(_.contains("scene(0) threw")) shouldBe Left(true)
    result.left.map(_.contains("Size must be positive")) shouldBe Left(true)

  // Review 2026-10-14 (menger#101): a static scene's failing `require` is an
  // ExceptionInInitializerError, which escaped the loader (live reload silently did nothing).
  it should "return Left, not throw, when a static scene's construction fails" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object FailingStaticScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(size = -1f)),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneLoader.load(file.getAbsolutePath)
    result.left.map(_.startsWith(SceneLoader.ConstructionFailed)) shouldBe Left(true)
    result.left.map(_.contains("Size must be positive")) shouldBe Left(true)

  it should "return Left, not throw, when a scene's construction recurses forever" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object RecursingScene:
        |  def deep(n: Int): Int = deep(n + 1) + 1
        |  val depth: Int = deep(0)
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere()),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    SceneLoader.load(file.getAbsolutePath).left.map(_.startsWith(SceneLoader.StackOverflow)) shouldBe
      Left(true)

  // Review 2026-10-14 (menger#79/#99): the compile view is an allow-list of the DSL surface, and
  // a compiled scene's class loader resolves nothing outside it at run time either.
  private val outsideTheVocabulary = List(
    "menger.tools.SceneValidator", "menger.engines.InteractiveEngine", "menger.cli.CliParser",
    "menger.scene.SceneCompiler", "menger.scene.SceneLoader", "menger.cli.MengerCLIOptions",
    "menger.config.SceneConfig", "examples.dsl.SceneIndex", "menger.app.Main"
  )

  it should "not compile a scene that names anything outside the DSL surface" in:
    outsideTheVocabulary.zipWithIndex.foreach { (name, i) =>
      val file = writeTempScene(s"object Reach$i:\n  val leak = classOf[$name]\n")
      withClue(name) { SceneCompiler.compile(file) shouldBe a[Left[?, ?]] }
    }

  "SceneClassLoader" should "resolve the platform and the DSL, nothing else" in:
    val loader = SceneClassLoader(getClass.getClassLoader)
    noException should be thrownBy loader.loadClass("menger.dsl.Sphere")
    noException should be thrownBy loader.loadClass("menger.spec.Projection4DSpec")
    noException should be thrownBy loader.loadClass("scala.collection.immutable.List")
    noException should be thrownBy loader.loadClass("java.lang.String")
    outsideTheVocabulary.foreach { name =>
      withClue(name) { an[ClassNotFoundException] should be thrownBy loader.loadClass(name) }
    }
