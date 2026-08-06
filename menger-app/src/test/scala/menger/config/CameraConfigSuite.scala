package menger.config

import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


class CameraConfigSuite extends AnyFlatSpec with Matchers:

  "CameraConfig case class" should "preserve all provided values" in:
    val pos = Vector[3](1f, 2f, 3f)
    val lookAt = Vector[3](4f, 5f, 6f)
    val up = Vector[3](0f, 1f, 0f)
    val config = CameraConfig(position = pos, lookAt = lookAt, up = up)

    config.position shouldBe pos
    config.lookAt shouldBe lookAt
    config.up shouldBe up

  "CameraConfig.Default" should "have camera at (0, 0, 3)" in:
    CameraConfig.Default.position(0) shouldBe 0f
    CameraConfig.Default.position(1) shouldBe 0f
    CameraConfig.Default.position(2) shouldBe 3f

  it should "look at origin (0, 0, 0)" in:
    CameraConfig.Default.lookAt(0) shouldBe 0f
    CameraConfig.Default.lookAt(1) shouldBe 0f
    CameraConfig.Default.lookAt(2) shouldBe 0f

  it should "have up vector pointing in +Y direction" in:
    CameraConfig.Default.up(0) shouldBe 0f
    CameraConfig.Default.up(1) shouldBe 1f
    CameraConfig.Default.up(2) shouldBe 0f

  // Edge case tests
  "CameraConfig edge cases" should "handle camera at origin" in:
    val config = CameraConfig(
      position = Vector.Zero[3],
      lookAt = Vector[3](0f, 0f, -1f),
      up = Vector[3](0f, 1f, 0f)
    )
    config.position shouldBe Vector.Zero[3]

  it should "handle camera looking at itself (degenerate case)" in:
    // This is mathematically invalid but should not throw
    val config = CameraConfig(
      position = Vector[3](1f, 1f, 1f),
      lookAt = Vector[3](1f, 1f, 1f),
      up = Vector[3](0f, 1f, 0f)
    )
    config.position shouldBe config.lookAt

  it should "handle non-normalized up vector" in:
    val config = CameraConfig(
      position = Vector.Zero[3],
      lookAt = Vector[3](0f, 0f, -1f),
      up = Vector[3](0f, 100f, 0f)  // Not normalized
    )
    config.up(1) shouldBe 100f

  it should "handle extreme coordinate values" in:
    val config = CameraConfig(
      position = Vector[3](1000000f, -1000000f, 1000000f),
      lookAt = Vector[3](-1000000f, 1000000f, -1000000f),
      up = Vector[3](0f, 1f, 0f)
    )
    config.position(0) shouldBe 1000000f
    config.lookAt(0) shouldBe -1000000f

  it should "handle tilted up vector" in:
    val config = CameraConfig(
      position = Vector.Zero[3],
      lookAt = Vector[3](0f, 0f, -1f),
      up = Vector[3](1f, 1f, 0f)  // 45-degree tilt
    )
    config.up(0) shouldBe 1f
    config.up(1) shouldBe 1f
