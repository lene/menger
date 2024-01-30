package menger

import org.scalatest.funsuite.AnyFunSuite
import com.badlogic.gdx.Version
import com.badlogic.gdx.backends.lwjgl3.{Lwjgl3Application, Lwjgl3ApplicationConfiguration}
import com.badlogic.gdx.graphics.GL20
import menger.objects.{Cube, Sphere}

class MengerSuite extends AnyFunSuite:
  test("libGDX version is high enough") {
      assert(Version.isHigherEqual(1, 12, 0))
  }

  test("instantiating a client works") {
    /**
     * Running this in sbt repeatedly causes:
     * java.lang.UnsatisfiedLinkError: Native Library /tmp/lwjgl{$USER}/.../liblwjgl.so already loaded in another classloader
     * So only run this with "sbt test"
     */
    Lwjgl3Application(EngineTest(0.1), Lwjgl3ApplicationConfiguration()).exit()
  }

  test("just instantiating a sphere does not store a model") {
    assert(Sphere.numStoredModels == 0)
    Sphere()
    assert(Sphere.numStoredModels == 0)
  }

  test("calling at() on a sphere stores model") {
    assert(Sphere.numStoredModels == 0)
    Sphere().at(0, 0, 0, 1)
    assert(Sphere.numStoredModels == 1)
  }

  test("calling at() on different spheres with same parameters stores only one model") {
    assert(Sphere.numStoredModels == 1)
    Sphere().at(0, 0, 0, 1)
    Sphere().at(0, 0, 0, 2)
    assert(Sphere.numStoredModels == 1)
  }

  test("calling at() on sphere with different parameters stores two models") {
    assert(Sphere.numStoredModels == 1)
    Sphere(divisions = 10).at(0, 0, 0, 1)
    assert(Sphere.numStoredModels == 2)
  }

  test("cube is one model") {
    assert(Cube().at(0, 0, 0, 1).size == 1)
  }

  test("cube model is stored") {
    assert(Cube.numStoredModels == 1)
  }

  test("different cube models are stored separately") {
    Cube(primitiveType = GL20.GL_LINES).at(0, 0, 0, 1)
    assert(Cube.numStoredModels == 2)
  }

  test("cube old_at() operates with six faces") {
    assert(Cube().old_at(0, 0, 0, 1).size == 6)
  }

  test("cube old_at() stores one square") {
    assert(Cube.numStoredFaces == 1)
  }

  test("sponge level 0 is one model") {
    assert(SpongeByVolume(0).at(0, 0, 0, 1).size == 1)
  }

  test("sponge level 1 has twenty times the size of level 0") {
    val cubeSize = SpongeByVolume(0).at(0, 0, 0, 1).size
    assert(SpongeByVolume(1).at(0, 0, 0, 1).size == 20 * cubeSize)
  }

  test("sponge level 2 has 400 times the size of level 0") {
    val cubeSize = SpongeByVolume(0).at(0, 0, 0, 1).size
    assert(SpongeByVolume(2).at(0, 0, 0, 1).size == 20 * 20 * cubeSize)
  }
