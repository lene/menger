package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneLoaderSuite extends AnyFlatSpec with Matchers:

  // SceneRegistry is a global, shared-JVM singleton (also populated by every example scene's
  // own class-loading side effects). `clear()` used to be called after each test here to tidy
  // up -- but sbt runs suites concurrently by default, so it raced with other suites reading
  // the registry (e.g. ExampleScenesSuite's "Scene registry" check), intermittently wiping
  // their view of it (usability review 2026-09 inbox item). This suite's keys ("test-scene",
  // "menger.dsl.TestSceneObject", "scene1"/"scene2"/"test") never collide with a real example
  // scene's registered short name, and this suite's own tests run in a fixed, sequential
  // order, so leaving them registered is safe -- clearing a registry every other suite shares
  // was never necessary for correctness here.
  "SceneLoader" should "load static scene from registry by name" in:
    val testScene = Scene(
      camera = Camera.Default,
      objects = List(Sphere(Material.Glass))
    )
    SceneRegistry.register("test-scene", testScene)

    val loaded = SceneLoader.load("test-scene")
    loaded shouldBe Right(LoadedScene.Static(testScene))

  it should "load static scene by fully-qualified class name" in:
    val loaded = SceneLoader.load("menger.dsl.TestSceneObject")
    loaded shouldBe a[Right[?, ?]]

    loaded match
      case Right(LoadedScene.Static(scene)) =>
        scene.objects should have length 1
        scene.objects.head shouldBe a[Sphere]
      case Right(LoadedScene.Animated(_)) =>
        fail("Expected Static but got Animated")
      case Left(error) =>
        fail(s"Failed to load scene: $error")

  it should "load animated scene by fully-qualified class name" in:
    val loaded = SceneLoader.load("menger.dsl.TestAnimatedSceneObject")
    loaded shouldBe a[Right[?, ?]]

    loaded match
      case Right(LoadedScene.Animated(fn)) =>
        val scene0 = fn(0f)
        scene0.objects should have length 1
        scene0.objects.head shouldBe a[Sphere]
        // At t=0, x=2*cos(0)=2, z=2*sin(0)=0
        scene0.objects.head match
          case s: Sphere =>
            s.pos.x shouldBe 2f +- 0.001f
            s.pos.z shouldBe 0f +- 0.001f
          case _ => fail("Expected Sphere")

        // At t=pi/2, x=2*cos(pi/2)~0, z=2*sin(pi/2)~2
        val scenePiHalf = fn(math.Pi.toFloat / 2f)
        scenePiHalf.objects.head match
          case s: Sphere =>
            s.pos.x shouldBe 0f +- 0.01f
            s.pos.z shouldBe 2f +- 0.01f
          case _ => fail("Expected Sphere")
      case Right(LoadedScene.Static(_)) =>
        fail("Expected Animated but got Static")
      case Left(error) =>
        fail(s"Failed to load scene: $error")

  it should "return different scenes for different t values" in:
    val loaded = SceneLoader.load("menger.dsl.TestAnimatedSceneObject")
    loaded match
      case Right(LoadedScene.Animated(fn)) =>
        val scene0 = fn(0f)
        val scene1 = fn(1f)
        (scene0.objects.head, scene1.objects.head) match
          case (s0: Sphere, s1: Sphere) =>
            s0.pos should not be s1.pos
          case _ => fail("Expected Spheres")
      case other =>
        fail(s"Expected Animated, got: $other")

  it should "prefer registry over reflection" in:
    val customScene = Scene(
      camera = Camera((1f, 2f, 3f), (0f, 0f, 0f)),
      objects = List(Cube(Material.Gold))
    )
    SceneRegistry.register("menger.dsl.TestSceneObject", customScene)

    val loaded = SceneLoader.load("menger.dsl.TestSceneObject")
    loaded shouldBe Right(LoadedScene.Static(customScene))

  it should "return error for non-existent scene" in:
    val loaded = SceneLoader.load("non-existent-scene")
    loaded shouldBe a[Left[?, ?]]

    loaded match
      case Left(error) =>
        error should include("Scene not found")
      case Right(_) =>
        fail("Should have failed to load non-existent scene")

  it should "return error for class without scene field or method" in:
    val loaded = SceneLoader.load("menger.dsl.SceneLoader")
    loaded shouldBe a[Left[?, ?]]

    loaded match
      case Left(error) =>
        error should include("has neither a 'scene' field")
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

  it should "allow overwriting registered scenes" in:
    val scene1 = Scene(Camera.Default, List(Sphere(Material.Glass)))
    val scene2 = Scene(Camera.Default, List(Cube(Material.Gold)))

    SceneRegistry.register("test", scene1)
    SceneRegistry.get("test") shouldBe Some(scene1)

    SceneRegistry.register("test", scene2)
    SceneRegistry.get("test") shouldBe Some(scene2)
