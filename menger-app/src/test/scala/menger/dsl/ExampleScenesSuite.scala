package menger.dsl

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

  private def extractStaticScene(result: Either[String, LoadedScene]): Scene =
    result match
      case Right(LoadedScene.Static(scene)) => scene
      case Right(LoadedScene.Animated(_)) => fail("Expected Static scene but got Animated")
      case Left(error) => fail(s"Failed to load scene: $error")

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

  private def extractAnimatedFn(result: Either[String, LoadedScene]): Float => Scene =
    result match
      case Right(LoadedScene.Animated(fn)) => fn
      case Right(LoadedScene.Static(_)) => fail("Expected Animated scene but got Static")
      case Left(error) => fail(s"Failed to load scene: $error")

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

  it should "load TesseractDemo via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.TesseractDemo"))
    scene.objects should have length 1
    scene.lights should have length 1

  it should "load MixedMetallicShowcase via reflection" in:
    val scene = extractStaticScene(SceneLoader.load("examples.dsl.MixedMetallicShowcase"))
    scene.objects should have length 5
    scene.lights should have length 1
    scene.planes should not be empty

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

  "SceneIndex" should "contain all static scenes" in:
    val all = examples.dsl.SceneIndex.all
    all should not be empty
    all.length should be >= 10

  it should "contain all animated scenes" in:
    val animated = examples.dsl.SceneIndex.animated
    animated should not be empty
