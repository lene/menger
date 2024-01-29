package menger

import org.scalatest.funsuite.AnyFunSuite
import com.badlogic.gdx.Version
import com.badlogic.gdx.backends.lwjgl3.{Lwjgl3Application, Lwjgl3ApplicationConfiguration}

class MengerSuite extends AnyFunSuite:
  test("libGDX version is high enough") {
      assert(Version.isHigherEqual(1, 12, 0))
  }

  test("instantiating a client works") {
    /**
     * Running this in sbt repeatedly causes:
     * java.lang.UnsatisfiedLinkError: Native Library /tmp/lwjgl{$USER}/.../liblwjgl.so already loaded in another classloader
     * So only run this with "sbt test"
     * Also does not yet have a way to stop the application.
     */
    new Lwjgl3Application(EngineTest(0.1), new Lwjgl3ApplicationConfiguration).exit()
  }

  test("cube has six faces") {
    val cube = new Cube()
    assert(cube.at(0, 0, 0, 1).size == 6)
  }

  test("sponge level 0 has six faces") {
    val sponge = new SpongeByVolume(0)
    assert(sponge.at(0, 0, 0, 1).size == 6)
  }

  test("sponge level 1 has twenty times six faces") {
    val sponge = new SpongeByVolume(1)
    assert(sponge.at(0, 0, 0, 1).size == 20 * 6)
  }
