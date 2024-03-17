package menger.objects

import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.backends.lwjgl3.{Lwjgl3Application, Lwjgl3ApplicationConfiguration}
import com.badlogic.gdx.graphics.{GL20, PerspectiveCamera}
import com.badlogic.gdx.math.Vector3
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.Tag
import org.scalamock.scalatest.MockFactory

import scala.runtime.stdLibPatches.Predef.assert
import menger.{MengerEngine, RotationProjectionParameters}
import menger.input.{EventDispatcher, CameraController, KeyController, Observer}


object GdxTest extends Tag("GdxTest")  // needs Gdx to be available

class GeometrySuite extends AnyFunSuite with MockFactory:
  // Can't mock Java class. Extend in Scala to mock: https://github.com/lampepfl/dotty/issues/18694
  class MockedCamera extends PerspectiveCamera
  private val camera = stub[MockedCamera]
  private val dispatcher = stub[EventDispatcher]

  private val ORIGIN = Vector3(0, 0, 0)
  private def controller = KeyController(camera, dispatcher)


  test("instantiating a client works", GdxTest) {
    /**
     * Running this in sbt repeatedly causes:
     * java.lang.UnsatisfiedLinkError: Native Library /tmp/lwjgl{$USER}/.../liblwjgl.so already loaded in another classloader
     * So only run this with "sbt test"
     */
    Lwjgl3Application(MengerEngine(0.01), Lwjgl3ApplicationConfiguration()).exit()
  }

  test("just instantiating a sphere does not store a model", GdxTest) {
    assert(Sphere.numStoredModels == 0)
    Sphere()
    assert(Sphere.numStoredModels == 0)
  }

  test("calling at() on a sphere stores model", GdxTest) {
    assert(Sphere.numStoredModels == 0)
    Sphere().at(ORIGIN, 1)
    assert(Sphere.numStoredModels == 1)
  }

  test("calling at() on different spheres stores only one model", GdxTest) {
    assert(Sphere.numStoredModels == 1)
    Sphere().at(ORIGIN, 1)
    Sphere().at(ORIGIN, 2)
    assert(Sphere.numStoredModels == 1)
  }

  test("calling at() on sphere with different parameters stores two models", GdxTest) {
    assert(Sphere.numStoredModels == 1)
    Sphere(divisions = 10).at(ORIGIN, 1)
    assert(Sphere.numStoredModels == 2)
  }

  test("sphere toString") {
    assert(Sphere().toString == "Sphere")
  }

  test("cube is one model", GdxTest) {
    assert(Cube().at(ORIGIN, 1).size == 1)
  }

  test("cube model is stored", GdxTest) {
    assert(Cube.numStoredModels == 1)
  }

  test("different cube models are stored separately", GdxTest) {
    Cube(primitiveType = GL20.GL_LINES).at(ORIGIN, 1)
    assert(Cube.numStoredModels == 2)
  }

  test("cube toString") {
    assert(Cube().toString == "Cube")
  }

  test("cube from squares operates with six faces", GdxTest) {
    assert(CubeFromSquares().at(ORIGIN, 1).size == 6)
  }

  test("cube from squares stores one square", GdxTest) {
    assert(CubeFromSquares.numStoredFaces == 1)
  }

  test("sponge level 0 is one model", GdxTest) {
    assert(SpongeByVolume(0).at(ORIGIN, 1).size == 1)
  }

  test("sponge level 1 has twenty times the size of level 0", GdxTest) {
    val cubeSize = SpongeByVolume(0).at(ORIGIN, 1).size
    assert(SpongeByVolume(1).at(ORIGIN, 1).size == 20 * cubeSize)
  }

  test("sponge level 2 has 400 times the size of level 0", GdxTest) {
    val cubeSize = SpongeByVolume(0).at(ORIGIN, 1).size
    assert(SpongeByVolume(2).at(ORIGIN, 1).size == 20 * 20 * cubeSize)
  }

  test("sponge by volume toString") {
    assert(SpongeByVolume(0).toString == "SpongeByVolume(level=0, 6 faces)")
  }

  test("sponge by surface toString") {
    assert(SpongeBySurface(0).toString == "SpongeBySurface(level=0, 6 faces)")
  }

  test("sponge by surface at() returns 6 faces regardless of level", GdxTest) {
    assert(SpongeBySurface(0).at(ORIGIN, 1).size == 6)
    assert(SpongeBySurface(1).at(ORIGIN, 1).size == 6)
  }

  test("face of sponge by surface level 1 has 12 subfaces") {
    assert(SpongeBySurface(1).faces.size == 12)
  }

  test("sponge by surface creates mesh(es)", GdxTest) {
    assert(SpongeBySurface(1).mesh.meshes.notEmpty)
  }

  test("MengerEngine with lines", GdxTest) {
    MengerEngine(0.01, lines = true).create()
  }

  test("MengerEngine with cube sponge", GdxTest) {
    MengerEngine(0.01, spongeType = "cube").create()
  }

  test("MengerEngine with tesseract", GdxTest) {
    MengerEngine(0.01, spongeType = "tesseract").create()
  }

  test("MengerEngine with invalid object type fails", GdxTest) {
    assertThrows[IllegalArgumentException] {
      MengerEngine(0.01, spongeType = "invalid").create()
    }
  }

  test("InputController should instantiate from a camera and dispatcher") {
    controller
  }

  private final val modKeys = Seq(
    Keys.CONTROL_LEFT, Keys.CONTROL_RIGHT, Keys.SHIFT_LEFT, Keys.SHIFT_RIGHT,
    Keys.ALT_LEFT, Keys.ALT_RIGHT
  )
  test("InputController.keyDown should recognize modifier keys") {
    modKeys.foreach { testKeyDown(controller, _) }
  }

  private final val rotateKeys = Seq(
    Keys.LEFT, Keys.RIGHT, Keys.UP, Keys.DOWN, Keys.PAGE_UP, Keys.PAGE_DOWN
  )
  test("InputController.keyDown should recognize rotate keys", GdxTest) {
    rotateKeys.foreach {testKeyDown(controller, _)}
  }

  test("rotate keys should rotate camera", GdxTest) {
    rotateKeys.foreach { testKeyDown(controller, _) }
    camera.rotateAround.verify(*, *, *).repeat(2 * rotateKeys.size)
  }

  test("InputController.keyDown should recognize Escape") {
    testKeyDown(controller, Keys.ESCAPE)
  }

  test("InputController.keyDown should recognize Q") {
    testKeyDown(controller, Keys.Q)
  }

  test("InputController.keyDown should not react to various other keys") {
    Seq(
      Keys.A, Keys.B, Keys.C, Keys.D, Keys.E, Keys.F, Keys.G, Keys.H, Keys.I, Keys.J, Keys.K,
      Keys.L, Keys.M, Keys.N, Keys.O, Keys.P, Keys.R, Keys.S, Keys.T, Keys.U, Keys.V, Keys.W,
      Keys.X, Keys.Y, Keys.Z
    ).foreach { key => assert(!controller.keyDown(key)) }
  }

  test("pressing shift should be recorded") {
    Seq(Keys.SHIFT_LEFT, Keys.SHIFT_RIGHT).foreach { key =>
      val thisController = controller
      thisController.keyDown(key)
      assert(thisController.shift)
    }
  }

  test("pressing ctrl should be recorded") {
    Seq(Keys.CONTROL_LEFT, Keys.CONTROL_RIGHT).foreach { key =>
      val thisController = controller
      thisController.keyDown(key)
      assert(thisController.ctrl)
    }
  }

  test("pressing alt should be recorded") {
    Seq(Keys.ALT_LEFT, Keys.ALT_RIGHT).foreach { key =>
      val thisController = controller
      thisController.keyDown(key)
      assert(thisController.alt)
    }
  }

  test("InputController should notify event dispatcher with shift pressed", GdxTest) {
    val thisController = controller
    thisController.keyDown(Keys.SHIFT_LEFT)
    thisController.keyDown(Keys.RIGHT)
    thisController.keyUp(Keys.SHIFT_LEFT)
    dispatcher.notifyObservers.verify(*).once()
  }

  test("EventDispatcher should notify observers with shift pressed", GdxTest) {
    class TestObserver extends Observer:
      var notified = false
      override def handleEvent(event: RotationProjectionParameters): Unit = notified = true

    val dispatcher = EventDispatcher()
    val observer = TestObserver()
    dispatcher.addObserver(observer)
    val thisController = KeyController(camera, dispatcher)
    thisController.keyDown(Keys.SHIFT_LEFT)
    thisController.keyDown(Keys.RIGHT)
    thisController.keyUp(Keys.SHIFT_LEFT)
    assert(observer.notified)
  }

  test("CameraInputController instantiated", GdxTest) {
    CameraController(camera, dispatcher)
  }

  test("CameraInputController touchDown", GdxTest) {
    val thisController = CameraController(camera, dispatcher)
    thisController.touchDown(0, 1, 0, 0)
  }

  test("CameraInputController touchDragged", GdxTest) {
    val thisController = CameraController(camera, dispatcher)
    thisController.touchDragged(0, 1, 0)
  }

  test("CameraInputController scrolled", GdxTest) {
    val thisController = CameraController(camera, dispatcher)
    thisController.scrolled(0, 1)
  }

  ignore("CameraInputController touchDragged with shift", GdxTest) {
    ???
    val thisController = CameraController(camera, dispatcher)
    thisController.touchDragged(0, 1, 0)
  }

def testKeyDown(inputController: KeyController, key: Int): Unit = {
  assert(!inputController.keyDown(key))
  inputController.keyUp(key)
}
