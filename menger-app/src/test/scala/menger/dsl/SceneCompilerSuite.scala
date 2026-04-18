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
