package menger

import org.scalatest.funsuite.AnyFunSuite
import com.badlogic.gdx.Version
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application

class MengerSuite extends AnyFunSuite {

  test("libGDX version is high enough") {
      assert(Version.isHigherEqual(1, 11, 0))
  }
  
  test("instantiating client works") {
    /**
     * Running this in sbt repeatedly causes:
     * java.lang.UnsatisfiedLinkError: Native Library /tmp/lwjgl{$USER}/.../liblwjgl.so already loaded in another classloader
     * So only run this with "sbt test"
     * Also does not yet have a way to stop the application.
     */
    val config = Main.config
    new Lwjgl3Application(new EngineTest(), config).exit()
  }

  test("cube has six faces") {
    val cube = new Cube()
    assert(cube.at(0, 0, 0, 1).size == 6)
  }
}
