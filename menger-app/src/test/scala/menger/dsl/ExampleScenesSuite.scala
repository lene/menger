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
