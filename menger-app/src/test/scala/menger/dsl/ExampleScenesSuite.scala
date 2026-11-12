package menger.dsl

import menger.common.ProfilingConfig
import menger.engines.scene.MeshFactory
import menger.scene.LoadError
import menger.scene.SceneLoader
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Integration tests for example DSL scenes.
 *
 * Verifies that all example scenes can be loaded and have valid
 * configurations. Does not render the scenes, just validates that
 * they compile and load correctly.
 */
class ExampleScenesSuite extends AnyFlatSpec with Matchers:

  // Force every scene object SceneIndex knows about to class-load (and therefore run its
  // SceneRegistry.register() call) before any test in this suite runs, so the "Scene
  // registry" check below doesn't depend on which "should load X via reflection" tests
  // happened to execute first (usability review 2026-09 inbox item: this test was
  // intermittently failing under certain `sbt testOnly` suite-glob combinations).
  examples.dsl.SceneIndex.all
  examples.dsl.SceneIndex.animated

  private def extractStaticScene(result: Either[LoadError, LoadedScene]): Scene =
    result match
      case Right(LoadedScene.Static(scene)) => scene
      case Right(LoadedScene.Animated(_)) => fail("Expected Static scene but got Animated")
      case Left(error) => fail(s"Failed to load scene: ${error.message}")

  "Example scenes" should "load GlassSphere via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.GlassSphere"))
    scene.objects should have length 1
    scene.lights should have length 1

  it should "load MengerShowcase via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.MengerShowcase"))
    scene.objects should have length 1
    scene.lights should have length 3
    scene.planes should not be empty

  it should "load SimpleScene via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.SimpleScene"))
    scene.objects should have length 1
    scene.lights should have length 1

  it should "load ThreeMaterials via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ThreeMaterials"))
    scene.objects should have length 3
    scene.lights should have length 2
    scene.planes should not be empty

  it should "load CausticsDemo via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.CausticsDemo"))
    scene.objects should have length 1
    scene.lights should have length 1
    scene.planes should not be empty
    scene.caustics shouldBe defined

  it should "load CustomMaterials via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.CustomMaterials"))
    scene.objects should have length 5
    scene.lights should have length 2
    scene.planes should not be empty

  it should "load ComplexLighting via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ComplexLighting"))
    scene.objects should have length 3
    scene.lights should have length 5
    scene.planes should not be empty

  it should "load SpongeShowcase via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.SpongeShowcase"))
    scene.objects should have length 3
    scene.lights should have length 2
    scene.planes should not be empty

  it should "load ReusableComponents via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ReusableComponents"))
    scene.objects should have length 4
    scene.lights should have length 3  // ThreePointLighting
    scene.planes should not be empty

  it should "load TrefoilKnot via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.TrefoilKnot"))
    scene.objects should have length 1
    scene.lights should have length 2
    scene.planes should not be empty

  private def extractAnimatedFn(result: Either[LoadError, LoadedScene]): Float => Scene =
    result match
      case Right(LoadedScene.Animated(fn)) => fn
      case Right(LoadedScene.Static(_)) => fail("Expected Animated scene but got Static")
      case Left(error) => fail(s"Failed to load scene: ${error.message}")

  "Animated example scenes" should "load OrbitingSphere as animated" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.OrbitingSphere"))
    for t <- List(0f, 0.5f, 1f) do
      val scene = fn(t)
      scene.objects should have length 1
      scene.lights should have length 1
      scene.planes should not be empty

  it should "load PulsingSponge as animated" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.PulsingSponge"))
    for t <- List(0f, 0.5f, 1f) do
      val scene = fn(t)
      scene.objects should have length 1
      scene.lights should have length 2
      scene.planes should not be empty

  it should "produce different OrbitingSphere scenes for different t" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.OrbitingSphere"))
    val scene0 = fn(0f)
    val scene1 = fn(1f)
    // Objects should have different positions
    scene0.objects.head should not be scene1.objects.head

  "Parametric example scenes" should "load ParametricSphere via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ParametricSphere"))
    scene.objects should have length 1
    scene.lights should have length 1
    scene.planes should not be empty

  it should "load ParametricTorus via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ParametricTorus"))
    scene.objects should have length 1
    scene.lights should have length 1
    scene.planes should not be empty

  it should "load ParametricWavySheet via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ParametricWavySheet"))
    scene.objects should have length 1
    scene.lights should have length 1
    scene.planes should not be empty

  it should "load ParametricMoebius via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ParametricMoebius"))
    scene.objects should have length 1
    scene.lights should have length 1

  it should "load ParametricKleinBottle via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ParametricKleinBottle"))
    scene.objects should have length 1
    scene.lights should have length 1

  it should "load ParametricKleinBottleFilm via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ParametricKleinBottleFilm"))
    scene.objects should have length 1
    scene.lights should have length 1

  it should "load ParametricSphereCaustics via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ParametricSphereCaustics"))
    scene.objects should have length 1
    scene.lights should have length 1
    scene.planes should not be empty
    scene.caustics shouldBe defined

  it should "load DenoiseIblDemo via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.DenoiseIblDemo"))
    scene.objects should have length 1
    scene.lights shouldBe empty
    scene.ibl shouldBe defined
    scene.ibl.get.samples shouldBe 1
    scene.render.map(_.accumulation) shouldBe Some(2)
    scene.render.map(_.denoise) shouldBe Some(DenoiseMode.Off)

  it should "load ParametricTorusCaustics via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.ParametricTorusCaustics"))
    scene.objects should have length 1
    scene.lights should have length 1
    scene.planes should not be empty
    scene.caustics shouldBe defined

  it should "load CausticsReferenceDefault via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.CausticsReferenceDefault"))
    scene.objects should have length 1
    scene.lights should have length 1
    scene.planes should not be empty
    scene.caustics shouldBe defined

  it should "load PolytopeGallery via reflection" in:
    // Usability review 2026-09, F17: the regular 4D polytopes besides the tesseract.
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.PolytopeGallery"))
    scene.objects should have length 5
    scene.lights should have length 1
    scene.planes should not be empty

  "Scene registry" should "have all registered short names" in:
    val registeredNames = SceneRegistry.list().sorted

    // Check that common scenes are registered
    registeredNames should contain("glass-sphere")
    registeredNames should contain("menger-showcase")
    registeredNames should contain("simple")
    registeredNames should contain("three-materials")
    registeredNames should contain("caustics-demo")
    registeredNames should contain("custom-materials")
    registeredNames should contain("complex-lighting")
    registeredNames should contain("sponge-showcase")
    registeredNames should contain("reusable-components")
    registeredNames should contain("parametric-sphere")
    registeredNames should contain("parametric-torus")
    registeredNames should contain("parametric-wavy-sheet")
    registeredNames should contain("parametric-moebius")
    registeredNames should contain("parametric-klein-bottle")
    registeredNames should contain("parametric-klein-bottle-film")
    registeredNames should contain("parametric-sphere-caustics")
    registeredNames should contain("parametric-torus-caustics")
    registeredNames should contain("caustics-reference-default")
    registeredNames should contain("denoise-ibl-demo")
    registeredNames should contain("trefoil-knot")
    registeredNames should contain("polytope-gallery")

  it should "load TesseractDemo via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.TesseractDemo"))
    scene.objects should have length 1
    scene.lights should have length 1

  it should "load MixedMetallicShowcase via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.MixedMetallicShowcase"))
    scene.objects should have length 5
    scene.lights should have length 1
    scene.planes should not be empty

  it should "load Plinth via reflection: a cylinder plinth under a glass sphere, caustics on" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.Plinth"))
    scene.objects.map(_.toObjectSpec.objectType) shouldBe List("cylinder", "sphere")
    scene.lights should have length 1
    scene.caustics shouldBe defined
    SceneRegistry.list() should contain("plinth")

  it should "load SunKeyLight via reflection: a Sun first, a glass cube on a plinth, caustics on" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.SunKeyLight"))
    scene.objects.map(_.toObjectSpec.objectType) shouldBe List("cylinder", "cube")
    scene.lights.head shouldBe a[Sun]
    scene.caustics shouldBe defined
    SceneRegistry.list() should contain("sun-key-light")

  it should "load FilmSphere via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.FilmSphere"))
    scene.objects should have length 3
    scene.lights should have length 2
    scene.planes should not be empty

  it should "load RotatingSilverSponge as animated" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.RotatingSilverSponge"))
    for t <- List(0f, 1.5f, 3f) do
      val scene = fn(t)
      scene.objects should have length 1
      scene.lights should have length 3
      scene.planes should not be empty

  it should "produce different RotatingSilverSponge scenes for different t" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.RotatingSilverSponge"))
    val scene0 = fn(0f)
    val scene3 = fn(3f)
    scene0.objects.head should not be scene3.objects.head

  it should "load TesseractToPenteract as an animated 5D tesseract (menger#162)" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.TesseractToPenteract"))
    for t <- List(0f, 2f, 4f, 6f, 8f) do
      val scene = fn(t)
      scene.objects should have length 1
      scene.lights should have length 2
      scene.planes should not be empty
      val spec = scene.objects.head.toObjectSpec
      spec.objectType shouldBe "tesseract"
      spec.projectionN.map(_.dimension) shouldBe Some(5)

  it should "turn TesseractToPenteract in xv, then yv, then zv, 45 degrees each" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.TesseractToPenteract"))
    def angles(t: Float) =
      val planes = fn(t).objects.head.toObjectSpec.projectionN.get.planes
      planes.map(r => (r.plane.letters, r.angle.value))
    def at(t: Float, expected: List[(String, Float)]) =
      val actual = angles(t)
      actual.map(_._1) shouldBe expected.map(_._1)
      actual.map(_._2).zip(expected.map(_._2)).foreach((a, e) => a shouldBe e +- 1e-3f)
    val duration = examples.dsl.TesseractToPenteract.durationSeconds
    val third = duration / 3f
    at(0f, List(("xv", 0f), ("yv", 0f), ("zv", 0f)))
    at(third, List(("xv", 45f), ("yv", 0f), ("zv", 0f)))
    at(2f * third, List(("xv", 45f), ("yv", 45f), ("zv", 0f)))
    at(duration, List(("xv", 45f), ("yv", 45f), ("zv", 45f)))

  it should "give TesseractToPenteract a strong 5D stage and the example's 4D stage" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.TesseractToPenteract"))
    val view = fn(0f).objects.head.toObjectSpec.projectionN.get
    view.stagePairs.map(s => (s.eye, s.screen)) shouldBe List((3f, 0.6f), (6f, 1.2f))

  it should "load HexeractTurning as an animated 6-cube with film faces, gold tubes (menger#163)" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.HexeractTurning"))
    for t <- List(0f, 2f, 4f, 8f) do
      val scene = fn(t)
      scene.objects should have length 1
      scene.lights should have length 2
      scene.planes should not be empty
      val spec = scene.objects.head.toObjectSpec
      (spec.objectType, spec.dimension) shouldBe (("n-cube", Some(6)))
      spec.material should not be empty
      spec.edgeRadius should not be empty
      spec.edgeMaterial should not be empty

  it should "turn HexeractTurning in the plane xu, a full turn over the loop" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.HexeractTurning"))
    def turns(t: Float) =
      val planes = fn(t).objects.head.toObjectSpec.projectionN.get.planes
      planes.map(r => (r.plane.letters, r.angle.value))
    val duration = examples.dsl.HexeractTurning.durationSeconds
    turns(0f) shouldBe List(("xu", 0f))
    turns(2f).head._2 shouldBe 45f +- 1e-3f
    turns(4f).head._2 shouldBe 90f +- 1e-3f
    val end = turns(duration)
    end.map(_._1) shouldBe List("xu")
    end.head._2 shouldBe 360f +- 1e-3f

  it should "close the HexeractTurning loop: the cube at the end is the cube at the start" in:
    // A quarter turn maps the shape onto itself but swaps faces of opposite winding, which a
    // film material shades differently; only a full turn returns every face to itself.
    given ProfilingConfig = ProfilingConfig.disabled
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.HexeractTurning"))
    def vertices(t: Float): Array[Float] =
      MeshFactory.create(fn(t).objects.head.toObjectSpec).vertices
    val start = vertices(0f)
    val end = vertices(examples.dsl.HexeractTurning.durationSeconds)
    end.length shouldBe start.length
    start.zip(end).foreach((a, b) => a shouldBe b +- 1e-5f)

  it should "fold PenteractFold out of the tesseract and back: scale 0 at the ends (menger#164)" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.PenteractFold"))
    val duration = examples.dsl.PenteractFold.durationSeconds
    def lastScale(t: Float): Option[Float] =
      fn(t).objects.head.toObjectSpec.scale.map(_.last)
    lastScale(0f) shouldBe Some(0f)
    lastScale(duration) shouldBe Some(0f)
    lastScale(duration / 2) shouldBe None
    lastScale(duration / 4).get shouldBe 0.5f +- 1e-5f
    val spec = fn(3f).objects.head.toObjectSpec
    (spec.objectType, spec.dimension) shouldBe (("n-cube", Some(5)))
    spec.edgeRadius should not be empty
    spec.material should not be empty

  it should "load PenteractSpongeBreath as an animated 5D surface sponge (menger#166)" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.PenteractSpongeBreath"))
    val duration = examples.dsl.PenteractSpongeBreath.durationSeconds
    duration shouldBe 16f
    for t <- List(0f, 2.6108146f, 4f, 8f, 12f) do
      val scene = fn(t)
      scene.objects should have length 1
      scene.lights should have length 2
      val spec = scene.objects.head.toObjectSpec
      (spec.objectType, spec.dimension, spec.maxCentreCoordinates) shouldBe
        (("n-sponge-surface", Some(5), Some(1)))
      spec.edgeRadius shouldBe None
      spec.material should not be empty

  // The level is 2 * smoothstep(t / 8) up to t = 8: 0.5 where 3x^2 - 2x^3 = 0.25 (x = t / 8),
  // which the freeze frames in integration-tests.sh and manual-test.sh use as t = 2.6108146.
  it should "have PenteractSpongeBreath at level 0.5 at the freeze-frame time" in:
    val x = (1 to 60).foldLeft((0.0, 1.0)) { case ((lo, hi), _) =>
      val mid = (lo + hi) / 2
      if 3 * mid * mid - 2 * mid * mid * mid < 0.25 then (mid, hi) else (lo, mid)
    }._1
    (x * examples.dsl.PenteractSpongeBreath.durationSeconds / 2).toFloat shouldBe 2.6108146f +- 1e-5f
    examples.dsl.PenteractSpongeBreath.level(2.6108146f) shouldBe 0.5f +- 1e-5f

  it should "breathe PenteractSpongeBreath from level 0 to 2 and back, chrome to grey plastic" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.PenteractSpongeBreath"))
    val duration = examples.dsl.PenteractSpongeBreath.durationSeconds
    def spec(t: Float) = fn(t).objects.head.toObjectSpec
    spec(0f).level shouldBe Some(0f)
    spec(duration).level shouldBe Some(0f)
    spec(duration / 2).level shouldBe Some(2f)
    spec(duration / 4).level.get shouldBe 1f +- 1e-5f
    spec(0f).material.get.metallic shouldBe 1f
    spec(duration / 2).material.get.metallic shouldBe 0f

  it should "keep PenteractSpongeBreath opaque, with no refraction, at every time" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.PenteractSpongeBreath"))
    (0 to 160).foreach { step =>
      val material = fn(step * 0.1f).objects.head.toObjectSpec.material.get
      material.ior shouldBe 1f
      material.color.a shouldBe 1f
    }

  it should "keep PenteractSpongeBreath within the level limit of a 5D surface sponge" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.PenteractSpongeBreath"))
    val limit = menger.dsl.ResourceLimits.ndSpongeLevel(
      menger.objects.higher_d.NdSpongeVariant.Surface, 5
    ).max
    limit should be >= 2
    (0 to 160).foreach { step =>
      val level = fn(step * 0.1f).objects.head.toObjectSpec.level.get
      level should (be >= 0f and be <= 2f)
    }
    fn(8f).objects.head.toObjectSpec.level.get should be <= limit.toFloat

  it should "turn PenteractSpongeBreath in xv and xz, a full turn over the loop" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.PenteractSpongeBreath"))
    val duration = examples.dsl.PenteractSpongeBreath.durationSeconds
    def turns(t: Float) =
      fn(t).objects.head.toObjectSpec.projectionN.get.planes.map(r =>
        (r.plane.letters, r.angle.value))
    turns(0f) shouldBe List(("xv", 0f), ("xz", 0f))
    turns(duration / 4).foreach((_, angle) => angle shouldBe 90f +- 1e-3f)
    turns(duration / 2).foreach((_, angle) => angle shouldBe 180f +- 1e-3f)
    // the loop wraps: the end of the loop is its start, one full turn later
    turns(duration).foreach((_, angle) => angle shouldBe 0f +- 1e-3f)
    turns(duration - 0.01f).foreach((_, angle) => angle should be > 359f)

  it should "repeat PenteractFold's extrusion outside the loop's first period" in:
    val extrusion = examples.dsl.PenteractFold.extrusion
    extrusion(18f) shouldBe extrusion(6f) +- 1e-5f
    extrusion(-3f) shouldBe extrusion(9f) +- 1e-5f
    extrusion(-3f) shouldBe 0.5f +- 1e-5f
    extrusion(-12f) shouldBe 0f +- 1e-5f

  it should "give PenteractFold's faces a thin film without refraction" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.PenteractFold"))
    fn(3f).objects.head.toObjectSpec.material.map(_.ior) shouldBe Some(1.0f)

  it should "close the PenteractFold loop: the same mesh at the start and the end" in:
    given ProfilingConfig = ProfilingConfig.disabled
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.PenteractFold"))
    def vertices(t: Float): Array[Float] =
      MeshFactory.create(fn(t).objects.head.toObjectSpec).vertices
    val start = vertices(0f)
    val end = vertices(examples.dsl.PenteractFold.durationSeconds)
    end.length shouldBe start.length
    start.zip(end).foreach((a, b) => a shouldBe b +- 1e-4f)

  "SceneIndex" should "contain all static scenes" in:
    val all = examples.dsl.SceneIndex.all
    all should not be empty
    all.length should be >= 10

  it should "contain all animated scenes" in:
    val animated = examples.dsl.SceneIndex.animated
    animated should not be empty

  it should "load DimensionLadder as animated, at the cycle length (menger#160)" in:
    val fn = extractAnimatedFn(SceneLoader.load("examples.dsl.DimensionLadder"))
    examples.dsl.DimensionLadder.durationSeconds shouldBe examples.dsl.DimensionLadder.cycleSeconds
    for t <- List(0f, 30f, 90f, 150f, 175f) do
      val scene = fn(t)
      scene.objects should have length 2
      scene.lights should have length 1
      scene.envMap should not be empty
