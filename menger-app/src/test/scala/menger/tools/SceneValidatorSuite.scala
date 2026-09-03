package menger.tools

import java.nio.file.Files

import menger.ObjectSpec
import menger.engines.scene.MeshFactory
import menger.objects.higher_d.PolytopeInvariants
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneValidatorSuite extends AnyFlatSpec with Matchers:

  private def writeTempScene(content: String): java.io.File =
    val f  = Files.createTempFile("scene-validator-test-", ".scala").toFile
    f.deleteOnExit()
    val pw = java.io.PrintWriter(f)
    pw.write(content)
    pw.close()
    f

  "SceneValidator.validate" should "return Ok for a valid, geometrically sound scene" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object OkScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere()),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.Ok
    result.messages shouldBe empty

  it should "return Ok for a valid scene containing a 4D object (exercises the geometric-invariant check end to end)" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object OkTesseractScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 2f, 5f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Tesseract(Material.Glass)),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.Ok

  it should "return CompileErrors naming the compiler's message for a Scala/DSL syntax error" in:
    val file = writeTempScene("object Broken { THIS IS NOT SCALA !!!!")
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.CompileErrors
    result.messages.head should include("Compilation of")

  it should "return Refused, never CompileErrors, when the scene file does not exist" in:
    val result = SceneValidator.validate(new java.io.File("/nonexistent/path/missing-scene.scala"))
    result.tag shouldBe SceneValidator.Tag.Refused

  it should "return Refused, not attempt to load, when the path is a directory rather than a file" in:
    val dir = Files.createTempDirectory("scene-validator-dir-test-").toFile
    dir.deleteOnExit()
    val result = SceneValidator.validate(dir)
    result.tag shouldBe SceneValidator.Tag.Refused

  // Review round 1 finding: RestrictedClasspath.preflight()'s failure branch was untested
  // through SceneValidator.validate -- RestrictedClasspathSuite tested preflight() in
  // isolation, and this file's own "file does not exist" test above exercises a *different*
  // Refused branch (the one above the preflight check in validate's own logic). The
  // production classpath is always structurally complete in this repo's dev/test
  // environment, so there was no way to reach this branch without the injectable seam below.
  it should "return Refused with the classpath reason, never attempting to load, when the restricted classpath preflight fails" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object NeverReachedScene:
        |  val scene: Scene = Scene(objects = List(Sphere()))
        |""".stripMargin
    )
    val result = SceneValidator.validate(file, preflightCheck = () => Some("forced failure for this test"))
    result.tag shouldBe SceneValidator.Tag.Refused
    result.messages.mkString should include("forced failure for this test")

  it should "catch a require() rejection cleanly (not a crash) and report it, not silently pass" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object InvalidSizeScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(size = -1f)),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.LintFindings
    result.messages.mkString should include("Size must be positive")

  "SceneValidator.run" should "report a usage error when given no arguments" in:
    SceneValidator.run(Array.empty) shouldBe a[Left[?, ?]]

  it should "delegate to validate() when given a scene file path" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object RunOkScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere()),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.run(Array(file.getAbsolutePath))
    result shouldBe a[Right[?, ?]]
    result.map(_.tag shouldBe SceneValidator.Tag.Ok)

  // Direct test of the reuse wiring SceneValidator's own `geometricFindings` uses internally
  // (SceneObject.toObjectSpec -> MeshFactory.mesh4D -> PolytopeInvariants.check), via public
  // APIs rather than SceneValidator's private methods: confirms a real DSL Tesseract, taken
  // through the same conversion the validator performs, produces a sound Mesh4D with no
  // findings -- and that MeshFactory.mesh4D's Option[Mesh4D] is None for a non-4D spec, which
  // is exactly the "when applicable" skip SceneValidator relies on for 3D objects.
  "the SceneObject -> ObjectSpec -> Mesh4D -> PolytopeInvariants chain SceneValidator reuses" should
    "produce no findings for a real Tesseract" in:
    val spec: ObjectSpec = menger.dsl.Tesseract(menger.dsl.Material.Glass).toObjectSpec
    val mesh = MeshFactory.mesh4D(spec)
    mesh shouldBe defined
    PolytopeInvariants.check(mesh.get) shouldBe empty

  it should "return None (not applicable) for a 3D object's ObjectSpec" in:
    val spec: ObjectSpec = menger.dsl.Sphere().toObjectSpec
    MeshFactory.mesh4D(spec) shouldBe None
