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

  "Example scenes" should "load GlassSphere via reflection" in:
    val result = SceneLoader.load("examples.dsl.GlassSphere")
    result shouldBe a[Right[?, ?]]
    result.foreach { scene =>
      scene.objects should have length 1
      scene.lights should have length 1
    }

  it should "load MengerShowcase via reflection" in:
    val result = SceneLoader.load("examples.dsl.MengerShowcase")
    result shouldBe a[Right[?, ?]]
    result.foreach { scene =>
      scene.objects should have length 1
      scene.lights should have length 3
      scene.plane shouldBe defined
    }

  it should "load SimpleScene via reflection" in:
    val result = SceneLoader.load("examples.dsl.SimpleScene")
    result shouldBe a[Right[?, ?]]
    result.foreach { scene =>
      scene.objects should have length 1
      scene.lights should have length 1
    }

  it should "load ThreeMaterials via reflection" in:
    val result = SceneLoader.load("examples.dsl.ThreeMaterials")
    result shouldBe a[Right[?, ?]]
    result.foreach { scene =>
      scene.objects should have length 3
      scene.lights should have length 2
      scene.plane shouldBe defined
    }

  it should "load CausticsDemo via reflection" in:
    val result = SceneLoader.load("examples.dsl.CausticsDemo")
    result shouldBe a[Right[?, ?]]
    result.foreach { scene =>
      scene.objects should have length 1
      scene.lights should have length 1
      scene.plane shouldBe defined
      scene.caustics shouldBe defined
    }

  it should "load CustomMaterials via reflection" in:
    val result = SceneLoader.load("examples.dsl.CustomMaterials")
    result shouldBe a[Right[?, ?]]
    result.foreach { scene =>
      scene.objects should have length 5
      scene.lights should have length 2
      scene.plane shouldBe defined
    }

  it should "load ComplexLighting via reflection" in:
    val result = SceneLoader.load("examples.dsl.ComplexLighting")
    result shouldBe a[Right[?, ?]]
    result.foreach { scene =>
      scene.objects should have length 3
      scene.lights should have length 5
      scene.plane shouldBe defined
    }

  it should "load SpongeShowcase via reflection" in:
    val result = SceneLoader.load("examples.dsl.SpongeShowcase")
    result shouldBe a[Right[?, ?]]
    result.foreach { scene =>
      scene.objects should have length 3
      scene.lights should have length 2
      scene.plane shouldBe defined
    }

  it should "load ReusableComponents via reflection" in:
    val result = SceneLoader.load("examples.dsl.ReusableComponents")
    result shouldBe a[Right[?, ?]]
    result.foreach { scene =>
      scene.objects should have length 4
      scene.lights should have length 3  // ThreePointLighting
      scene.plane shouldBe defined
    }

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
