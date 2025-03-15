package menger.objects

import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.backends.lwjgl3.{Lwjgl3Application, Lwjgl3ApplicationConfiguration}
import com.badlogic.gdx.graphics.{GL20, PerspectiveCamera}
import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.Tag
import org.scalamock.scalatest.MockFactory

import menger.{MengerEngine, RotationProjectionParameters}
import menger.input.{CameraController, EventDispatcher, KeyController, Observer}
import org.scalatest.matchers.should.Matchers


object GdxTest extends Tag("GdxTest")  // needs Gdx to be available

class GeometrySuite extends AnyFlatSpec with MockFactory with Matchers:
  // Can't mock Java class. Extend in Scala to mock: https://github.com/lampepfl/dotty/issues/18694
  class MockedCamera extends PerspectiveCamera
  private val camera = stub[MockedCamera]
  private val dispatcher = stub[EventDispatcher]

  private val ORIGIN = Vector3(0, 0, 0)
  private def controller = KeyController(camera, dispatcher)
  private lazy val loadingLWJGLSucceeds: Boolean =
    try
      Lwjgl3Application(MengerEngine(0.01), Lwjgl3ApplicationConfiguration()).exit()
      true
    catch
      case _: UnsatisfiedLinkError => false

  "instantiating a client" should "work" taggedAs GdxTest in:
    /**
     * Running this in sbt repeatedly causes:
     * java.lang.UnsatisfiedLinkError: Native Library /tmp/lwjgl{$USER}/.../liblwjgl.so already loaded in another classloader
     * So only run this with "sbt test"
     */
    assume(loadingLWJGLSucceeds)
    loadingLWJGLSucceeds should be (true)

  "instantiating a sphere" should "not store a model" taggedAs GdxTest in:
    Sphere.numStoredModels should be (0)
    Sphere()
    Sphere.numStoredModels should be(0)

  "calling at() on a sphere" should "store a model" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Sphere.numStoredModels should be(0)
    Sphere().at(ORIGIN, 1)
    Sphere.numStoredModels should be(1)

  "calling at() on different spheres" should "store only one model" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Sphere.numStoredModels should be(1)
    Sphere().at(ORIGIN, 1)
    Sphere().at(ORIGIN, 2)
    Sphere.numStoredModels should be(1)

  "calling at() on sphere with different parameters" should "store two models" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Sphere.numStoredModels should be(1)
    Sphere(divisions = 10).at(ORIGIN, 1)
    Sphere.numStoredModels should be(2)

  "sphere toString" should "return class name" in:
    Sphere().toString should be("Sphere")

  "cube" should "be one model" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Cube().at(ORIGIN, 1) should have size 1

  it should "store one model" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Cube.numStoredModels should be(1)

  "different cube models" should "be stored separately" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Cube(primitiveType = GL20.GL_LINES).at(ORIGIN, 1)
    Cube.numStoredModels should be(2)

  "cube toString" should "return class name" in:
    Cube().toString should be("Cube")

  "cube from squares" should "operate with six faces" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    CubeFromSquares().at(ORIGIN, 1) should have size 6

  it should "store one square" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    CubeFromSquares.numStoredFaces should be(1)

  "sponge level 0" should "be one model" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    SpongeByVolume(0).at(ORIGIN, 1) should have size 1

  "sponge level 1" should "have twenty times the size of level 0" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    val cubeSize = SpongeByVolume(0).at(ORIGIN, 1).size
    SpongeByVolume(1).at(ORIGIN, 1) should have size 20 * cubeSize

  "sponge level 2" should "have 400 times the size of level 0" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    val cubeSize = SpongeByVolume(0).at(ORIGIN, 1).size
    SpongeByVolume(2).at(ORIGIN, 1) should have size 20 * 20 * cubeSize

  "sponge by volume toString" should "return class name" in:
    SpongeByVolume(0).toString should be("SpongeByVolume(level=0, 6 faces)")

  "sponge by surface" should "have toString return class name" in:
    assume(loadingLWJGLSucceeds)
    SpongeBySurface(0).toString should be("SpongeBySurface(level=0, 6 faces)")

  it should "have at() returns 6 faces regardless of level" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    SpongeBySurface(0).at(ORIGIN, 1) should have size 6
    SpongeBySurface(1).at(ORIGIN, 1) should have size 6

  it should "create mesh(es)" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    SpongeBySurface(1).mesh.meshes should not be empty

  "face of sponge by surface level 1" should "have 12 subfaces" in:
    SpongeBySurface(1).faces should have size 12

  "MengerEngine" should "instantiate with lines" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    MengerEngine(0.01, lines = true).create()

  it should "instantiate with cube sponge" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    MengerEngine(0.01, spongeType = "cube").create()

  it should "instantiate with tesseract" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    MengerEngine(0.01, spongeType = "tesseract").create()

  it should "fail with invalid object type" taggedAs GdxTest in:
    an [IllegalArgumentException] should be thrownBy MengerEngine(0.01, spongeType = "invalid").create()

  "InputController" should "instantiate from a camera and dispatcher" in:
    assume(loadingLWJGLSucceeds)
    controller

  it should "notify event dispatcher with shift pressed" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    val thisController = controller
    thisController.keyDown(Keys.SHIFT_LEFT)
    thisController.keyDown(Keys.RIGHT)
    thisController.keyUp(Keys.SHIFT_LEFT)
    dispatcher.notifyObservers.verify(*).once()

  private final val modKeys = Seq(
    Keys.CONTROL_LEFT, Keys.CONTROL_RIGHT, Keys.SHIFT_LEFT, Keys.SHIFT_RIGHT,
    Keys.ALT_LEFT, Keys.ALT_RIGHT
  )
  private final val rotateKeys = Seq(
    Keys.LEFT, Keys.RIGHT, Keys.UP, Keys.DOWN, Keys.PAGE_UP, Keys.PAGE_DOWN
  )
  "InputController.keyDown" should "recognize modifier keys" in:
    assume(loadingLWJGLSucceeds)
    modKeys.foreach { testKeyDown(controller, _) }

  it should "recognize rotate keys" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    rotateKeys.foreach {testKeyDown(controller, _)}

  it should "recognize Escape" in:
    testKeyDown(controller, Keys.ESCAPE)

  it should "recognize Q" in:
    testKeyDown(controller, Keys.Q)

  it should "not react to various other keys" in:
    Seq(
      Keys.A, Keys.B, Keys.C, Keys.D, Keys.E, Keys.F, Keys.G, Keys.H, Keys.I, Keys.J, Keys.K,
      Keys.L, Keys.M, Keys.N, Keys.O, Keys.P, Keys.R, Keys.S, Keys.T, Keys.U, Keys.V, Keys.W,
      Keys.X, Keys.Y, Keys.Z
    ).foreach { key => !controller.keyDown(key) should be (true) }

  "rotate keys" should "rotate camera" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    rotateKeys.foreach {testKeyDown(controller, _)}
    camera.rotateAround.verify(*, *, *).repeat(2 * rotateKeys.size)

  "pressing shift" should "be recorded" in:
    Seq(Keys.SHIFT_LEFT, Keys.SHIFT_RIGHT).foreach { key =>
      val thisController = controller
      thisController.keyDown(key)
      thisController.shift should be (true)
    }

  "pressing ctrl" should "be recorded" in:
    Seq(Keys.CONTROL_LEFT, Keys.CONTROL_RIGHT).foreach { key =>
      val thisController = controller
      thisController.keyDown(key)
      thisController.ctrl should be (true)
    }

  "pressing alt" should "be recorded" in:
    Seq(Keys.ALT_LEFT, Keys.ALT_RIGHT).foreach { key =>
      val thisController = controller
      thisController.keyDown(key)
      thisController.alt should be (true)
    }

  "EventDispatcher" should "notify observers with shift pressed" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
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
    observer.notified should be (true)

  "CameraInputController" should "instantiate" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    CameraController(camera, dispatcher)

  it should "record touchDown" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    val thisController = CameraController(camera, dispatcher)
    thisController.touchDown(0, 1, 0, 0)

  it should "record touchDragged" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    val thisController = CameraController(camera, dispatcher)
    thisController.touchDragged(0, 1, 0)

  it should "record scrolled" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    val thisController = CameraController(camera, dispatcher)
    thisController.scrolled(0, 1)

  ignore should "record touchDragged with shift" taggedAs GdxTest in:
    ???
    val thisController = CameraController(camera, dispatcher)
    thisController.touchDragged(0, 1, 0)

  def testKeyDown(inputController: KeyController, key: Int): Unit = {
    inputController.keyDown(key) should be (false)
    inputController.keyUp(key)
  }
