package menger.dsl

import java.nio.file.Files

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

  it should "return Left for a file with a syntax error" in:
    val file = writeTempScene("object Broken { THIS IS NOT SCALA !!!!")
    SceneCompiler.compile(file) shouldBe a[Left[?, ?]]

  // RestrictedClasspath's own doc comment (menger.dsl.RestrictedClasspath): omitting
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

  it should "read an animated scene's declared duration in seconds" in:
    val file = writeTempScene(animatedScene("TimedScene", "val duration = 10f", "1f + t"))
    SceneLoader.load(file.getAbsolutePath) match
      case Right(animated: LoadedScene.Animated) => animated.duration shouldBe Some(10f)
      case other => fail(s"Expected Animated, got $other")

  it should "leave the duration empty when an animated scene declares none" in:
    val file = writeTempScene(animatedScene("UntimedScene", "", "1f + t"))
    SceneLoader.load(file.getAbsolutePath) match
      case Right(animated: LoadedScene.Animated) => animated.duration shouldBe None
      case other => fail(s"Expected Animated, got $other")

  it should "reject a non-positive duration" in:
    val file = writeTempScene(animatedScene("NegativeDurationScene", "val duration = -1f", "1f + t"))
    SceneLoader.load(file.getAbsolutePath) shouldBe Left("'duration' must be positive, got -1.0")

  // Usability review 2026-09: the loader swallowed exceptions from probing scene(0), so a scene
  // whose require() failed at t=0 was reported as having no scene method at all.
  it should "report the scene's own error when scene(0) throws" in:
    val file = writeTempScene(animatedScene("ThrowingAtZeroScene", "", "t"))
    val result = SceneLoader.load(file.getAbsolutePath)
    result.left.map(_.contains("scene(0) threw")) shouldBe Left(true)
    result.left.map(_.contains("Size must be positive")) shouldBe Left(true)
