package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.language.implicitConversions

class CameraSuite extends AnyFlatSpec with Matchers:

  "Camera" should "have correct defaults" in:
    val camera = new Camera()
    camera.position shouldBe Vec3(0f, 0f, 3f)
    camera.lookAt shouldBe Vec3.Zero
    camera.up shouldBe Vec3(0f, 1f, 0f)

  it should "be constructible with position only" in:
    val camera = new Camera(position = Vec3(1f, 2f, 3f))
    camera.position shouldBe Vec3(1f, 2f, 3f)
    camera.lookAt shouldBe Vec3.Zero
    camera.up shouldBe Vec3(0f, 1f, 0f)

  it should "be constructible with position and lookAt" in:
    val camera = new Camera(position = Vec3(1f, 2f, 3f), lookAt = Vec3(4f, 5f, 6f))
    camera.position shouldBe Vec3(1f, 2f, 3f)
    camera.lookAt shouldBe Vec3(4f, 5f, 6f)
    camera.up shouldBe Vec3(0f, 1f, 0f)

  it should "be constructible with position, lookAt, and up" in:
    val camera = new Camera(Vec3(1f, 2f, 3f), Vec3(4f, 5f, 6f), Vec3(0f, 0f, 1f))
    camera.position shouldBe Vec3(1f, 2f, 3f)
    camera.lookAt shouldBe Vec3(4f, 5f, 6f)
    camera.up shouldBe Vec3(0f, 0f, 1f)

  it should "accept Float tuples for both position and lookAt" in:
    val camera = Camera((1f, 2f, 3f), (4f, 5f, 6f))
    camera.position shouldBe Vec3(1f, 2f, 3f)
    camera.lookAt shouldBe Vec3(4f, 5f, 6f)

  it should "accept Float tuples for position, lookAt, and up" in:
    val camera = Camera((1f, 2f, 3f), (4f, 5f, 6f), (0f, 0f, 1f))
    camera.position shouldBe Vec3(1f, 2f, 3f)
    camera.lookAt shouldBe Vec3(4f, 5f, 6f)
    camera.up shouldBe Vec3(0f, 0f, 1f)

  it should "accept Int tuples for both position and lookAt" in:
    val camera = Camera((1, 2, 3), (4, 5, 6))
    camera.position shouldBe Vec3(1f, 2f, 3f)
    camera.lookAt shouldBe Vec3(4f, 5f, 6f)

  it should "accept Double tuples for both position and lookAt" in:
    val camera = Camera((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    camera.position shouldBe Vec3(1f, 2f, 3f)
    camera.lookAt shouldBe Vec3(4f, 5f, 6f)

  "Camera.Default" should "have correct configuration" in:
    val camera = Camera.Default
    camera.position shouldBe Vec3(0f, 0f, 3f)
    camera.lookAt shouldBe Vec3.Zero
    camera.up shouldBe Vec3(0f, 1f, 0f)

  "Camera.toCameraConfig" should "convert to CameraConfig correctly" in:
    val camera = Camera(Vec3(1f, 2f, 3f), Vec3(4f, 5f, 6f), Vec3(0f, 1f, 0f))
    val config = camera.toCameraConfig

    config.position.x shouldBe 1f
    config.position.y shouldBe 2f
    config.position.z shouldBe 3f
    config.lookAt.x shouldBe 4f
    config.lookAt.y shouldBe 5f
    config.lookAt.z shouldBe 6f
    config.up.x shouldBe 0f
    config.up.y shouldBe 1f
    config.up.z shouldBe 0f

  it should "convert default camera correctly" in:
    val camera = Camera.Default
    val config = camera.toCameraConfig

    config.position.x shouldBe 0f
    config.position.y shouldBe 0f
    config.position.z shouldBe 3f
    config.lookAt.x shouldBe 0f
    config.lookAt.y shouldBe 0f
    config.lookAt.z shouldBe 0f
