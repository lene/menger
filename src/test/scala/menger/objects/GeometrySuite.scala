package menger.objects

import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.backends.lwjgl3.{Lwjgl3Application, Lwjgl3ApplicationConfiguration}
import com.badlogic.gdx.graphics.{GL20, PerspectiveCamera}
import com.badlogic.gdx.math.Vector3
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.Tag

import scala.runtime.stdLibPatches.Predef.assert
import menger.MengerEngine
import menger.input.{InputController, InputController3D}


object GfxTest extends Tag("GfxTest")

class GeometrySuite extends AnyFunSuite:
  private val ORIGIN = Vector3(0, 0, 0)
  private lazy val camera = PerspectiveCamera(67, 10, 10)


  test("instantiating a client works", GfxTest) {
    /**
     * Running this in sbt repeatedly causes:
     * java.lang.UnsatisfiedLinkError: Native Library /tmp/lwjgl{$USER}/.../liblwjgl.so already loaded in another classloader
     * So only run this with "sbt test"
     */
    Lwjgl3Application(MengerEngine(0.1), Lwjgl3ApplicationConfiguration()).exit()
  }

  test("just instantiating a sphere does not store a model", GfxTest) {
    assert(Sphere.numStoredModels == 0)
    Sphere()
    assert(Sphere.numStoredModels == 0)
  }

  test("calling at() on a sphere stores model", GfxTest) {
    assert(Sphere.numStoredModels == 0)
    Sphere().at(ORIGIN, 1)
    assert(Sphere.numStoredModels == 1)
  }

  test("calling at() on different spheres stores only one model", GfxTest) {
    assert(Sphere.numStoredModels == 1)
    Sphere().at(ORIGIN, 1)
    Sphere().at(ORIGIN, 2)
    assert(Sphere.numStoredModels == 1)
  }

  test("calling at() on sphere with different parameters stores two models", GfxTest) {
    assert(Sphere.numStoredModels == 1)
    Sphere(divisions = 10).at(ORIGIN, 1)
    assert(Sphere.numStoredModels == 2)
  }

  test("cube is one model", GfxTest) {
    assert(Cube().at(ORIGIN, 1).size == 1)
  }

  test("cube model is stored", GfxTest) {
    assert(Cube.numStoredModels == 1)
  }

  test("different cube models are stored separately", GfxTest) {
    Cube(primitiveType = GL20.GL_LINES).at(ORIGIN, 1)
    assert(Cube.numStoredModels == 2)
  }

  test("cube from squares operates with six faces", GfxTest) {
    assert(CubeFromSquares().at(ORIGIN, 1).size == 6)
  }

  test("cube from squares stores one square", GfxTest) {
    assert(CubeFromSquares.numStoredFaces == 1)
  }

  test("sponge level 0 is one model", GfxTest) {
    assert(SpongeByVolume(0).at(ORIGIN, 1).size == 1)
  }

  test("sponge level 1 has twenty times the size of level 0", GfxTest) {
    val cubeSize = SpongeByVolume(0).at(ORIGIN, 1).size
    assert(SpongeByVolume(1).at(ORIGIN, 1).size == 20 * cubeSize)
  }

  test("sponge level 2 has 400 times the size of level 0", GfxTest) {
    val cubeSize = SpongeByVolume(0).at(ORIGIN, 1).size
    assert(SpongeByVolume(2).at(ORIGIN, 1).size == 20 * 20 * cubeSize)
  }

  test("sphere toString") {
    assert(Sphere().toString == "Sphere")
  }

  test("cube toString") {
    assert(Cube().toString == "Cube")
  }

  test("sponge by volume toString") {
    assert(SpongeByVolume(0).toString == "SpongeByVolume(level=0, 6 faces)")
  }

  test("sponge by surface toString") {
    assert(SpongeBySurface(0).toString == "SpongeBySurface(level=0, 6 faces)")
  }

  test("sponge by surface at() returns 6 faces regardless of level", GfxTest) {
    assert(SpongeBySurface(0).at(ORIGIN, 1).size == 6)
    assert(SpongeBySurface(1).at(ORIGIN, 1).size == 6)
  }

  test("face of sponge by surface level 1 has 12 subfaces") {
    assert(SpongeBySurface(1).faces.size == 12)
  }

  test("sponge by surface creates mesh(es)", GfxTest) {
    assert(SpongeBySurface(1).mesh.meshes.notEmpty)
  }

  test("InputController3D should instantiate from a camera", GfxTest) {
    InputController3D(camera)
  }

  test("InputController3D.keyDown should recognize CTRL", GfxTest) {
    val inputController = InputController3D(camera)
    Seq(
      Keys.CONTROL_LEFT, Keys.CONTROL_RIGHT
    ).foreach { testKeyDown(inputController, _) }
  }

  test("InputController3D.keyDown should recognize arrow keys", GfxTest) {
    val inputController = InputController3D(camera)
    Seq(
      Keys.LEFT, Keys.RIGHT, Keys.UP, Keys.DOWN
    ).foreach { testKeyDown(inputController, _) }
  }

  test("InputController3D.keyDown should recognize Escape", GfxTest) {
    val inputController = InputController3D(camera)
    testKeyDown(inputController, Keys.ESCAPE)
  }


  test("InputController3D.keyDown should recognize Q", GfxTest) {
    val inputController = InputController3D(camera)
    testKeyDown(inputController, Keys.Q)
  }

  test("InputController3D.keyDown should not recognize various other keys", GfxTest) {
    val inputController = InputController3D(camera)
    Seq(
      Keys.A, Keys.B, Keys.C, Keys.D, Keys.E, Keys.F, Keys.G, Keys.H, Keys.I, Keys.J,
      Keys.K, Keys.L, Keys.M, Keys.N, Keys.O, Keys.P, Keys.R, Keys.S, Keys.T, Keys.U,
      Keys.V, Keys.W, Keys.X, Keys.Y, Keys.Z
    ).foreach { key =>
      assert(!inputController.keyDown(key))
    }
  }


def testKeyDown(inputController: InputController, key: Int): Unit = {
  assert(inputController.keyDown(key))
  inputController.keyUp(key)
}