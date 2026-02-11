package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneLoaderSuite extends AnyFlatSpec with Matchers:

  "SceneLoader" should "load scene from registry by name" in:
    val testScene = Scene(
      camera = Camera.Default,
      objects = List(Sphere(Material.Glass))
    )
    SceneRegistry.register("test-scene", testScene)

    val loaded = SceneLoader.load("test-scene")
    loaded shouldBe Right(testScene)

    SceneRegistry.clear()

  it should "load scene by fully-qualified class name" in:
    // Create a test scene object
    val loaded = SceneLoader.load("menger.dsl.TestSceneObject")
    loaded shouldBe a[Right[?, ?]]

    loaded match
      case Right(scene) =>
        scene.objects should have length 1
        scene.objects.head shouldBe a[Sphere]
      case Left(error) =>
        fail(s"Failed to load scene: $error")

  it should "prefer registry over reflection" in:
    val customScene = Scene(
      camera = Camera((1f, 2f, 3f), (0f, 0f, 0f)),
      objects = List(Cube(Material.Gold))
    )
    SceneRegistry.register("menger.dsl.TestSceneObject", customScene)

    val loaded = SceneLoader.load("menger.dsl.TestSceneObject")
    loaded shouldBe Right(customScene)

    SceneRegistry.clear()

  it should "return error for non-existent scene" in:
    val loaded = SceneLoader.load("non-existent-scene")
    loaded shouldBe a[Left[?, ?]]

    loaded match
      case Left(error) =>
        error should include("Scene not found")
      case Right(_) =>
        fail("Should have failed to load non-existent scene")

  it should "return error for class without scene field" in:
    val loaded = SceneLoader.load("menger.dsl.SceneLoader")
    loaded shouldBe a[Left[?, ?]]

    loaded match
      case Left(error) =>
        error should include("does not have a 'scene' field")
      case Right(_) =>
        fail("Should have failed to load class without scene field")

  "SceneRegistry" should "register and retrieve scenes" in:
    val scene1 = Scene(Camera.Default, List(Sphere(Material.Glass)))
    val scene2 = Scene(Camera.Default, List(Cube(Material.Gold)))

    SceneRegistry.register("scene1", scene1)
    SceneRegistry.register("scene2", scene2)

    SceneRegistry.get("scene1") shouldBe Some(scene1)
    SceneRegistry.get("scene2") shouldBe Some(scene2)
    SceneRegistry.get("non-existent") shouldBe None

    SceneRegistry.list() should contain allOf ("scene1", "scene2")

    SceneRegistry.clear()

  it should "allow overwriting registered scenes" in:
    val scene1 = Scene(Camera.Default, List(Sphere(Material.Glass)))
    val scene2 = Scene(Camera.Default, List(Cube(Material.Gold)))

    SceneRegistry.register("test", scene1)
    SceneRegistry.get("test") shouldBe Some(scene1)

    SceneRegistry.register("test", scene2)
    SceneRegistry.get("test") shouldBe Some(scene2)

    SceneRegistry.clear()
