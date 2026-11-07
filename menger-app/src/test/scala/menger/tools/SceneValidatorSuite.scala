package menger.tools

import java.nio.file.Files

import menger.spec.ObjectSpec
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

  // Usability review 2026-09: only t=0 was checked, but a scene that changes with t is often
  // at its largest (or broken) at the end of its declared duration.
  // Review 2026-10-14, menger#80: two level-5 volume sponges each pass their own cap and ran
  // the render window out of memory together.
  it should "reject a scene over the scene-wide budget as a lint finding" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object BudgetBombScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(
        |      Sponge(spongeType = SpongeType.VolumeFilling, level = 5f),
        |      Sponge(spongeType = SpongeType.VolumeFilling, level = 5f, pos = (2f, 0f, 0f))),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.LintFindings
    result.messages.exists(_.startsWith("scene-budget: scene over budget")) shouldBe true

  it should "check an animated scene with a declared duration at its end time too" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object ShrinksToNothingScene:
        |  val durationSeconds = 2f
        |  def scene(t: Float): Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(size = 1f - t)),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.LintFindings
    result.messages.exists(_.startsWith("scene-evaluation: scene(2.0) threw")) shouldBe true

  // Usability review 2026-10, session 4 (F94): an edge-rendered tesseract sponge rising to
  // level 3 and back passed, because only t = 0 and t = durationSeconds (both level 0) were
  // checked; level 3 with edges needs more instances than the renderer has.
  it should "reject a scene that is only invalid in the middle of its time range" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object PeaksMidAnimationScene:
        |  val durationSeconds = 12f
        |  def scene(t: Float): Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 6f), lookAt = (0f, 0f, 0f)),
        |    objects = List(TesseractSponge(
        |      spongeType = SurfaceSubdividing,
        |      level = 3f * (1f - math.abs(t - 6f) / 6f),
        |      edgeRadius = Some(0.02f)
        |    )),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.LintFindings
    result.messages.exists(_.contains("instances")) shouldBe true

  // Usability review 2026-10, session 4 (F90, F89, F102): the scene agent reads the evaluated
  // values from here instead of parsing scene text.
  "SceneValidator snapshots" should "carry a static scene's evaluated objects, camera and lights" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object SnapshotScene:
        |  private val Lift = 1.5f
        |  private val Shiny = Material.Glass
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 1f, 6f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(pos = (0f, Lift * 2f, 0f), material = Shiny, size = 0.5f)),
        |    lights  = List(Directional(direction = (1f, -1f, -1f), intensity = 1.5f))
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.Ok
    result.snapshots should have size 1
    val snapshot = result.snapshots.head
    snapshot.camera.position shouldBe List(0f, 1f, 6f)
    val sphere = snapshot.objects.head
    sphere.objectType shouldBe "sphere"
    sphere.pos shouldBe List(0f, 3f, 0f)
    sphere.material.map(_.ior) shouldBe Some(1.5f)
    snapshot.lights.head.kind shouldBe "Directional"
    snapshot.lights.head.intensity shouldBe 1.5f
    result.cameraPath shouldBe empty

  // Review 2026-10-14 (menger#92): planes were missing, so a plane change never reached the
  // agent's change list; objects carried only the renderer's name. In a node tree the DSL name
  // must pair with its own flattened spec.
  it should "carry planes and each object's DSL type, in a node tree too" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object TreeScene:
        |  val scene: Scene = Scene(
        |    root = Some(SceneNode(
        |      geometry = Some(Cube(pos = (1f, 0f, 0f))),
        |      children = List(
        |        SceneNode(geometry = Some(Sponge(VolumeFilling, level = 1f))),
        |        SceneNode(geometry = Some(Sphere(pos = (0f, 2f, 0f))))
        |      )
        |    )),
        |    planes = List(Plane(Y at -1, color = "#336699"))
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.Ok
    val snapshot = result.snapshots.head
    snapshot.objects.map(o => (o.dslType, o.objectType)) shouldBe
      List(("Cube", "cube"), ("Sponge", "sponge-volume"), ("Sphere", "sphere"))
    snapshot.objects(2).pos shouldBe List(0f, 2f, 0f)
    snapshot.planes should have size 1
    val plane = snapshot.planes.head
    (plane.axis, plane.value) shouldBe ("Y", -1f)
    plane.color.map(_.take(3)) shouldBe Some(List(0x33 / 255f, 0x66 / 255f, 0x99 / 255f))
    plane.checkerColor shouldBe None

  // menger#162: the nD view of a 4D object is in the snapshot, in the order written.
  it should "carry a ProjectionSpec's dimension, planes and stages" in:
    val file = writeTempScene(
      """import menger.dsl._
        |import menger.spec.{PlaneRotation, ProjectionSpec}
        |object NdScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 2f, 5f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Tesseract(
        |      material = Some(Material.Glass),
        |      projection = Some(ProjectionSpec(
        |        5, planes = List(PlaneRotation("xv", 30f), PlaneRotation("wv", 10f)),
        |        eye = 4f, screen = 1f))
        |    )),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.Ok
    val tesseract = result.snapshots.head.objects.head
    tesseract.projection shouldBe None
    val nd = tesseract.ndProjection.get
    nd.dimension shouldBe 5
    nd.planes.map(p => (p.plane, p.angle)) shouldBe List(("xv", 30f), ("wv", 10f))
    nd.stages shouldBe List.fill(2)(SceneSnapshot.StageValues(4f, 1f))

  it should "carry an animated scene's snapshots and its densely sampled camera path" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object CameraFlightScene:
        |  val durationSeconds = 2f
        |  def scene(t: Float): Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 6f - t), lookAt = (0f, 0f, -t)),
        |    objects = List(Sphere()),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.Ok
    result.snapshots should have size SceneValidator.AnimationSampleCount.toLong
    result.snapshots.last.camera.position shouldBe List(0f, 0f, 4f)
    result.cameraPath.size shouldBe (2f / SceneSnapshot.CameraPathStepSeconds).round + 1
    result.cameraPath.head.position shouldBe List(0f, 0f, 6f)
    result.cameraPath.last.t shouldBe 2f

  "SceneValidator.sampleTimes" should "span the whole duration, both ends included" in:
    val times = SceneValidator.sampleTimes(Some(12f))
    times should have size SceneValidator.AnimationSampleCount.toLong
    times.head shouldBe 0f
    times.last shouldBe 12f
    times should contain (6f)
    SceneValidator.sampleTimes(None) shouldBe List(0f)

  // Usability review 2026-09 (T1#1): an absolute or `..`-escaping texture path is a
  // security/config error, flagged before the scene reaches a renderer with a real
  // `--texture-dir`.
  it should "flag an absolute texture path" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object AbsoluteTexturePathScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(texture = Some("/etc/passwd"))),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.LintFindings
    result.messages.exists(_.startsWith("asset-path:")) shouldBe true

  it should "flag a texture path that escapes the texture directory with .." in:
    val file = writeTempScene(
      """import menger.dsl._
        |object EscapingTexturePathScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(texture = Some("../../etc/passwd"))),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.LintFindings
    result.messages.exists(_.startsWith("asset-path:")) shouldBe true

  it should "accept a plain relative texture path" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object RelativeTexturePathScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(texture = Some("brick.png"))),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.Ok

  // Usability review 2026-09 (T3#13): a degenerate free-form/lambda object used to pass
  // straight through -- require() alone can't catch a bad closure's behavior at the seam.
  it should "flag a ParametricSurface whose closedU seam doesn't actually coincide" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object BadSeamScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(ParametricSurface(f = (u, v) => Vec3(u, 0f, v), closedU = true)),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.LintFindings
    result.messages.exists(_.startsWith("parametric-surface-seam:")) shouldBe true

  it should "flag a Curve whose control points all coincide" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object DegenerateCurveScene:
        |  val p = Vec3(1f, 2f, 3f)
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Curve(points = Seq(p, p, p, p))),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.LintFindings
    result.messages.exists(_.startsWith("curve-arc-length:")) shouldBe true

  // Usability review 2026-09 (F18/F19/F24): scenes the validator accepted crashed the render
  // window -- it never ran the renderer's own grouping and builder preconditions.
  private def sceneOf(objects: menger.dsl.SceneObject*): menger.dsl.Scene =
    menger.dsl.Scene(
      camera = menger.dsl.Camera(
        position = menger.dsl.Vec3(0f, 0f, 5f), lookAt = menger.dsl.Vec3(0f, 0f, 0f)
      ),
      objects = objects.toList,
      lights = List()
    )

  private val edgedSponge = menger.dsl.TesseractSponge(
    spongeType = menger.dsl.TesseractSpongeType.VolumeRemoving,
    level = 0f,
    material = Some(menger.dsl.Material.Film),
    edgeRadius = Some(0.005f)
  )

  "SceneValidator.buildFindings" should "accept a 4D object with edges next to one without" in:
    val plain = menger.dsl.Tesseract(pos = menger.dsl.Vec3(0f, 2f, 0f))
    SceneValidator.buildFindings(sceneOf(edgedSponge, plain)) shouldBe empty

  it should "accept a sphere next to an edge-rendered 4D object" in:
    val orb = menger.dsl.Sphere(size = 0.6f)
    SceneValidator.buildFindings(sceneOf(edgedSponge, orb)) shouldBe empty

  // Usability review 2026-09, session 2 (F44, menger#52): this used to be rejected; each 4D
  // object is projected with its own `projection`.
  it should "accept edge-rendered 4D objects whose 4D projections differ" in:
    val otherProjection = menger.dsl.Tesseract(
      pos = menger.dsl.Vec3(0f, 2f, 0f),
      edgeRadius = Some(0.005f),
      projection = Some(menger.spec.Projection4DSpec(eyeW = 5f))
    )
    SceneValidator.buildFindings(sceneOf(edgedSponge, otherProjection)) shouldBe empty

  // Usability review 2026-09, session 2 (F33, menger#53): counting an edge-rendered sponge's
  // edges built its whole mesh, which exhausted the validator's memory from level 3 up. Those
  // levels need more edge cylinders than the instance limit allows; say so without building.
  it should "reject edge-rendered tesseract sponges from level 3 up without building them" in:
    for
      (spongeType, levels) <- List(
        menger.dsl.TesseractSpongeType.VolumeRemoving -> List(3f, 4f),
        menger.dsl.TesseractSpongeType.SurfaceSubdividing -> List(3f, 5f)
      )
      level <- levels
    do
      val sponge = edgedSponge.copy(spongeType = spongeType, level = level)
      withClue(s"$spongeType level $level: ") {
        val findings = SceneValidator.buildFindings(sceneOf(sponge))
        findings.map(_.invariant) shouldBe List("scene-build")
        findings.head.message should include("instances")
      }

  it should "still accept an edge-rendered surface sponge at level 2" in:
    val sponge = edgedSponge.copy(
      spongeType = menger.dsl.TesseractSpongeType.SurfaceSubdividing, level = 2f
    )
    SceneValidator.buildFindings(sceneOf(sponge)) shouldBe empty

  it should "find nothing to object to in any of the renderer's own registered example scenes" in:
    val _ = examples.dsl.SceneIndex
    menger.dsl.SceneRegistry.list().foreach { name =>
      withClue(s"example scene '$name': ") {
        SceneValidator.buildFindings(menger.dsl.SceneRegistry.get(name).get) shouldBe empty
      }
    }

  // Review round 2: the only 4D scene exercised here was a default-size `Tesseract`, whose 16
  // vertices are all equidistant from the origin -- so `common-sphere` passed and nothing
  // noticed that `MeshFactory.mesh4D` also routes the 4D *fractals* into a check written for
  // regular polytopes. A sponge at level >= 1 has vertex norms spanning [0.33, 1.0] and was
  // reported `lint-findings`, including for the renderer's own shipped corpus exemplars.
  it should "return Ok for a volume-removing tesseract sponge at level 1 (a fractal, not a regular polytope)" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object SpongeScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 2f, 5f), lookAt = (0f, 0f, 0f)),
        |    objects = List(TesseractSponge(
        |      spongeType = TesseractSpongeType.VolumeRemoving, level = 1f
        |    )),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.Ok

  it should "return Ok for a surface-subdividing tesseract sponge at level 1" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object Sponge2Scene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 2f, 5f), lookAt = (0f, 0f, 0f)),
        |    objects = List(TesseractSponge(
        |      spongeType = TesseractSpongeType.SurfaceSubdividing, level = 1f, size = 1.5f
        |    )),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.Ok

  // Usability review 2026-09, session 2 (F31): the geometric check built the full 4D mesh of
  // every sponge just to check its vertices, so any sponge from level 3 up (level 4 has ~127M
  // faces) ran the sandboxed validator out of heap and came back `refused`.
  // Was the DSL's maximum level (4). Level 4 is about 42M faces (measured growth x38 per level:
  // 24, 768, 29184), over the scene budget (menger#80), so the highest level that can pass is 3.
  it should "return Ok for a tesseract sponge at the highest level within the scene budget" in:
    val file = writeTempScene(
      s"""import menger.dsl._
        |object MaxLevelSpongeScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 2f, 5f), lookAt = (0f, 0f, 0f)),
        |    objects = List(TesseractSponge(
        |      spongeType = TesseractSpongeType.VolumeRemoving,
        |      level = ${menger.common.Const.Engine.tesseractSpongeMaxLevel - 1}f
        |    )),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.Ok

  // SA-AD-4 rule 2: a compiled scene sees the DSL surface and its transitive needs, not the rest
  // of menger-app. `menger.tools`/`menger.engines`/`menger.cli`/`menger.input` are pruned off
  // the classpath handed to the compiler (review round 2), so referencing them is a compile
  // error rather than a working import.
  it should "refuse a scene that reaches outside the DSL surface into menger.tools" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object ReachesOutside:
        |  val leak = menger.tools.SceneValidator.SchemaVersion
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere()),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    SceneValidator.validate(file).tag shouldBe SceneValidator.Tag.CompileErrors

  // The scene-graph authoring style: `Scene` requires `objects` or `root`, so for a root-only
  // scene the whole geometric check hangs off the `root` half of `geometricFindings`. Nothing
  // covered it, and deleting that half left every test green (review round 2).
  it should "run the geometric check on a scene-graph scene's leaf geometry, not just the flat objects list" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object GraphScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 2f, 5f), lookAt = (0f, 0f, 0f)),
        |    root   = SceneNode(geometry = Some(Tesseract(Material.Glass))),
        |    lights = List()
        |  )
        |""".stripMargin
    )
    SceneValidator.validate(file).tag shouldBe SceneValidator.Tag.Ok

  // `def scene(t: Float)` is a first-class DSL form and roughly half the shipped corpus uses
  // it, but every scene in this suite declared `val scene`, so the Animated branch was never
  // reached (review round 2).
  it should "validate an animated scene through the Animated branch" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object AnimatedScene:
        |  def scene(t: Float): Scene = Scene(
        |    camera = Camera(position = (0f, 2f, 5f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Tesseract(Material.Glass)),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    SceneValidator.validate(file).tag shouldBe SceneValidator.Tag.Ok

  it should "return CompileErrors naming the compiler's message for a Scala/DSL syntax error" in:
    val file = writeTempScene("object Broken { THIS IS NOT SCALA !!!!")
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.CompileErrors
    result.messages.head should include("Compilation of")

  // menger#113: the agent repairs from the error's kind and line, not from the message text.
  it should "carry the kind and the line of a compile error in loadError" in:
    val file = writeTempScene("import menger.dsl._\nobject OnLineTwo:\n  val scene: Scene = 42\n")
    val failure = SceneValidator.validate(file).loadError
    failure.map(_.kind) shouldBe Some("compile_failed")
    failure.toList.flatMap(_.diagnostics).map(_.line) should contain (3)

  it should "carry a loadError of a different kind for a scene that loads but fails" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object FailingScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere(size = -1f)),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.LintFindings
    result.loadError.map(_.kind) shouldBe Some("construction_failed")

  it should "leave loadError empty for a scene that loads" in:
    SceneValidator.validate(writeTempScene(
      """import menger.dsl._
        |object FineScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Sphere()),
        |    lights  = List()
        |  )
        |""".stripMargin
    )).loadError shouldBe None

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

  // menger#156: optix-jni shades a cylinder opaque, so glass on one would silently render
  // opaque; the validator says so instead.
  "SceneValidator.opaqueOnlyFindings" should "report a Cylinder with a transparent material" in:
    for material <- List(menger.dsl.Material.Glass, menger.dsl.Material.Film) do
      val findings = SceneValidator.opaqueOnlyFindings(
        sceneOf(menger.dsl.Cylinder(material = Some(material)))
      )
      withClue(material) {
        findings.map(_.invariant) shouldBe List(SceneValidator.OpaqueOnlyInvariant)
        findings.head.message should include("object 0, a Cylinder at pos (0.0, 0.0, 0.0)")
        findings.head.message should include("renders opaque")
      }

  it should "report a Cylinder whose colour alone is transparent" in:
    val cylinder = menger.dsl.Cylinder(color = Some(menger.dsl.Color(1f, 0f, 0f, 0.5f)))
    SceneValidator.opaqueOnlyFindings(sceneOf(cylinder)) should have size 1

  // Caustics photons deposit only on IOR 1.05 or less: an opaque Plastic plinth (1.5) would
  // silently lose its caustic.
  it should "report a Cylinder with a refractive IOR, opaque or not, as getting no caustic" in:
    for cylinder <- List(
      menger.dsl.Cylinder(material = Some(menger.dsl.Material.Plastic)),
      menger.dsl.Cylinder(ior = 1.5f)
    ) do
      val findings = SceneValidator.opaqueOnlyFindings(sceneOf(cylinder))
      withClue(cylinder) {
        findings.map(_.invariant) shouldBe List(SceneValidator.OpaqueOnlyInvariant)
        findings.head.message should include("receives no caustic")
        findings.head.message should not include "transparency"
      }

  it should "name each object by its index and place, so equal ones stay apart" in:
    val glass = menger.dsl.Cylinder(material = Some(menger.dsl.Material.Glass))
    val placed = glass.copy(
      p0 = Some(menger.dsl.Vec3(0f, -1f, 0f)), p1 = Some(menger.dsl.Vec3(0f, 1f, 0f))
    )
    val messages = SceneValidator.opaqueOnlyFindings(sceneOf(glass, glass, placed)).map(_.message)
    messages should have size 3
    messages(0) should include("object 0,")
    messages(1) should include("object 1,")
    messages(2) should include("object 2, a Cylinder at p0 (0.0,-1.0,0.0), p1 (0.0,1.0,0.0)")

  it should "accept an opaque Cylinder and a glass object of another type" in:
    SceneValidator.opaqueOnlyFindings(sceneOf(
      menger.dsl.Cylinder(material = Some(menger.dsl.Material.Matte)),
      menger.dsl.Cylinder(material = Some(menger.dsl.Material.Chrome)),
      menger.dsl.Sphere(material = Some(menger.dsl.Material.Glass)),
      menger.dsl.Cone(material = Some(menger.dsl.Material.Glass))
    )) shouldBe empty

  it should "make a scene with a glass Cylinder a lint finding" in:
    val file = writeTempScene(
      """import menger.dsl._
        |object GlassCylinderScene:
        |  val scene: Scene = Scene(
        |    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
        |    objects = List(Cylinder(material = Some(Material.Glass))),
        |    lights  = List()
        |  )
        |""".stripMargin
    )
    val result = SceneValidator.validate(file)
    result.tag shouldBe SceneValidator.Tag.LintFindings
    result.findings.map(_.invariant) shouldBe List(SceneValidator.OpaqueOnlyInvariant)

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
