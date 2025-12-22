package menger.input

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SphericalOrbitSuite extends AnyFlatSpec with Matchers:

  // Concrete implementation for testing the trait
  class TestOrbit(config: OrbitConfig = OrbitConfig()) extends SphericalOrbit:
    override protected def orbitConfig: OrbitConfig = config

    @SuppressWarnings(Array("org.wartremover.warts.Var"))
    private var _azimuth: Float = 0f
    @SuppressWarnings(Array("org.wartremover.warts.Var"))
    private var _elevation: Float = 0f
    @SuppressWarnings(Array("org.wartremover.warts.Var"))
    private var _distance: Float = 1f

    override protected def azimuth: Float = _azimuth
    override protected def azimuth_=(value: Float): Unit = _azimuth = value
    override protected def elevation: Float = _elevation
    override protected def elevation_=(value: Float): Unit = _elevation = value
    override protected def distance: Float = _distance
    override protected def distance_=(value: Float): Unit = _distance = value

    // Public accessors for testing
    def getAzimuth: Float = _azimuth
    def setAzimuth(value: Float): Unit = _azimuth = value
    def getElevation: Float = _elevation
    def setElevation(value: Float): Unit = _elevation = value
    def getDistance: Float = _distance
    def setDistance(value: Float): Unit = _distance = value

    // Expose protected methods for testing
    def testInitSpherical(eye: Vector3, lookAt: Vector3): (Float, Float, Float) =
      initSpherical(eye, lookAt)

    def testUpdateOrbit(deltaX: Int, deltaY: Int): Unit =
      updateOrbit(deltaX, deltaY)

    def testUpdateZoom(scrollAmount: Float): Unit =
      updateZoom(scrollAmount)

    def testSphericalToCartesian(lookAt: Vector3): Vector3 =
      sphericalToCartesian(lookAt)

    def testComputePanOffset(deltaX: Int, deltaY: Int, forward: Vector3, up: Vector3): Vector3 =
      computePanOffset(deltaX, deltaY, forward, up)

  "initSpherical" should "compute correct spherical coords for camera on Z axis" in:
    val orbit = TestOrbit()
    val eye = Vector3(0, 0, 3)
    val lookAt = Vector3(0, 0, 0)
    val (azimuth, elevation, distance) = orbit.testInitSpherical(eye, lookAt)

    distance shouldBe (3f +- 0.001f)
    azimuth shouldBe (0f +- 0.1f)
    elevation shouldBe (0f +- 0.1f)

  it should "compute correct spherical coords for camera on X axis" in:
    val orbit = TestOrbit()
    val eye = Vector3(3, 0, 0)
    val lookAt = Vector3(0, 0, 0)
    val (azimuth, elevation, distance) = orbit.testInitSpherical(eye, lookAt)

    distance shouldBe (3f +- 0.001f)
    azimuth shouldBe (90f +- 0.1f)
    elevation shouldBe (0f +- 0.1f)

  it should "compute correct spherical coords for camera above origin" in:
    val orbit = TestOrbit()
    val eye = Vector3(0, 3, 0)
    val lookAt = Vector3(0, 0, 0)
    val (azimuth, elevation, distance) = orbit.testInitSpherical(eye, lookAt)

    distance shouldBe (3f +- 0.001f)
    elevation shouldBe (90f +- 0.1f)

  it should "compute correct spherical coords for offset lookAt" in:
    val orbit = TestOrbit()
    val eye = Vector3(5, 0, 0)
    val lookAt = Vector3(2, 0, 0)
    val (azimuth, elevation, distance) = orbit.testInitSpherical(eye, lookAt)

    distance shouldBe (3f +- 0.001f)

  "updateOrbit" should "increase azimuth with positive deltaX" in:
    val orbit = TestOrbit()
    orbit.testUpdateOrbit(10, 0)
    orbit.getAzimuth shouldBe (3f +- 0.1f)  // 10 * 0.3 sensitivity

  it should "decrease elevation with positive deltaY (inverted for natural feel)" in:
    val orbit = TestOrbit()
    orbit.testUpdateOrbit(0, 10)
    orbit.getElevation shouldBe (-3f +- 0.1f)  // -10 * 0.3 sensitivity

  it should "clamp elevation to prevent gimbal lock" in:
    val orbit = TestOrbit()
    // Try to go way past max elevation
    orbit.testUpdateOrbit(0, -1000)
    orbit.getElevation shouldBe (89f +- 0.1f)  // Clamped to maxElevation

    // Try to go way past min elevation
    orbit.testUpdateOrbit(0, 2000)
    orbit.getElevation shouldBe (-89f +- 0.1f)  // Clamped to minElevation

  it should "accumulate multiple orbit updates" in:
    val orbit = TestOrbit()
    orbit.testUpdateOrbit(10, 0)
    orbit.testUpdateOrbit(10, 0)
    orbit.getAzimuth shouldBe (6f +- 0.1f)

  "updateZoom" should "increase distance with positive scroll" in:
    val orbit = TestOrbit()
    val initialDistance = orbit.getDistance
    orbit.testUpdateZoom(1f)
    orbit.getDistance should be > initialDistance

  it should "decrease distance with negative scroll" in:
    val orbit = TestOrbit()
    orbit.setDistance(5f)
    val initialDistance = orbit.getDistance
    orbit.testUpdateZoom(-1f)
    orbit.getDistance should be < initialDistance

  it should "clamp distance to minimum" in:
    val orbit = TestOrbit()
    orbit.setDistance(1f)
    // Zoom in aggressively
    (1 to 20).foreach(_ => orbit.testUpdateZoom(-1f))
    orbit.getDistance shouldBe (0.12f +- 0.05f)  // minDistance

  it should "clamp distance to maximum" in:
    val orbit = TestOrbit()
    orbit.setDistance(10f)
    // Zoom out aggressively
    (1 to 20).foreach(_ => orbit.testUpdateZoom(1f))
    orbit.getDistance shouldBe (20f +- 0.01f)  // maxDistance

  "sphericalToCartesian" should "return eye on Z axis for azimuth=0, elevation=0" in:
    val orbit = TestOrbit()
    orbit.setAzimuth(0f)
    orbit.setElevation(0f)
    orbit.setDistance(3f)
    val eye = orbit.testSphericalToCartesian(Vector3(0, 0, 0))

    eye.x shouldBe (0f +- 0.001f)
    eye.y shouldBe (0f +- 0.001f)
    eye.z shouldBe (3f +- 0.001f)

  it should "return eye on X axis for azimuth=90" in:
    val orbit = TestOrbit()
    orbit.setAzimuth(90f)
    orbit.setElevation(0f)
    orbit.setDistance(3f)
    val eye = orbit.testSphericalToCartesian(Vector3(0, 0, 0))

    eye.x shouldBe (3f +- 0.001f)
    eye.y shouldBe (0f +- 0.001f)
    eye.z shouldBe (0f +- 0.001f)

  it should "return eye above origin for elevation=90" in:
    val orbit = TestOrbit()
    orbit.setAzimuth(0f)
    orbit.setElevation(90f)
    orbit.setDistance(3f)
    val eye = orbit.testSphericalToCartesian(Vector3(0, 0, 0))

    eye.x shouldBe (0f +- 0.001f)
    eye.y shouldBe (3f +- 0.001f)
    eye.z shouldBe (0f +- 0.001f)

  it should "offset by lookAt position" in:
    val orbit = TestOrbit()
    orbit.setAzimuth(0f)
    orbit.setElevation(0f)
    orbit.setDistance(3f)
    val eye = orbit.testSphericalToCartesian(Vector3(1, 2, 3))

    eye.x shouldBe (1f +- 0.001f)
    eye.y shouldBe (2f +- 0.001f)
    eye.z shouldBe (6f +- 0.001f)  // 3 + 3

  "computePanOffset" should "produce right vector for positive deltaX" in:
    val orbit = TestOrbit()
    orbit.setDistance(1f)
    val forward = Vector3(0, 0, -1)
    val up = Vector3(0, 1, 0)
    val offset = orbit.testComputePanOffset(100, 0, forward, up)

    // Right vector should be positive X for this forward/up
    offset.x should be > 0f
    offset.y shouldBe (0f +- 0.01f)

  it should "produce up vector for negative deltaY" in:
    val orbit = TestOrbit()
    orbit.setDistance(1f)
    val forward = Vector3(0, 0, -1)
    val up = Vector3(0, 1, 0)
    val offset = orbit.testComputePanOffset(0, -100, forward, up)

    // Negative deltaY (mouse up) should produce positive Y offset
    offset.x shouldBe (0f +- 0.01f)
    offset.y should be > 0f

  it should "scale with distance" in:
    val orbit1 = TestOrbit()
    orbit1.setDistance(1f)
    val offset1 = orbit1.testComputePanOffset(100, 0, Vector3(0, 0, -1), Vector3(0, 1, 0))

    val orbit2 = TestOrbit()
    orbit2.setDistance(2f)
    val offset2 = orbit2.testComputePanOffset(100, 0, Vector3(0, 0, -1), Vector3(0, 1, 0))

    offset2.len() shouldBe ((offset1.len() * 2f) +- 0.01f)

  "OrbitConfig" should "have sensible defaults" in:
    val config = OrbitConfig()
    config.orbitSensitivity shouldBe 0.3f
    config.panSensitivity shouldBe 0.005f
    config.zoomSensitivity shouldBe 0.1f
    config.minDistance shouldBe 0.1f
    config.maxDistance shouldBe 20.0f
    config.minElevation shouldBe -89.0f
    config.maxElevation shouldBe 89.0f

  it should "allow custom values" in:
    val config = OrbitConfig(
      orbitSensitivity = 0.5f,
      panSensitivity = 0.01f,
      zoomSensitivity = 0.2f,
      minDistance = 1f,
      maxDistance = 100f,
      minElevation = -45f,
      maxElevation = 45f
    )
    config.orbitSensitivity shouldBe 0.5f
    config.maxDistance shouldBe 100f

  "Custom orbit config" should "affect orbit sensitivity" in:
    val orbit = TestOrbit(OrbitConfig(orbitSensitivity = 1.0f))
    orbit.testUpdateOrbit(10, 0)
    orbit.getAzimuth shouldBe (10f +- 0.1f)  // 10 * 1.0 sensitivity

  it should "affect elevation limits" in:
    val orbit = TestOrbit(OrbitConfig(maxElevation = 45f))
    orbit.testUpdateOrbit(0, -500)  // Try to go above limit
    orbit.getElevation shouldBe (45f +- 0.1f)

  it should "affect zoom limits" in:
    val orbit = TestOrbit(OrbitConfig(maxDistance = 5f))
    orbit.setDistance(4f)
    (1 to 10).foreach(_ => orbit.testUpdateZoom(1f))
    orbit.getDistance shouldBe (5f +- 0.1f)
