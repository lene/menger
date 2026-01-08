package menger.config

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


class CameraConfigSuite extends AnyFlatSpec with Matchers:

  "CameraConfig case class" should "preserve all provided values" in:
    val pos = Vector3(1f, 2f, 3f)
    val lookAt = Vector3(4f, 5f, 6f)
    val up = Vector3(0f, 1f, 0f)
    val config = CameraConfig(position = pos, lookAt = lookAt, up = up)
    
    config.position shouldBe pos
    config.lookAt shouldBe lookAt
    config.up shouldBe up

  "CameraConfig.Default" should "have camera at (0, 0, 3)" in:
    CameraConfig.Default.position.x shouldBe 0f
    CameraConfig.Default.position.y shouldBe 0f
    CameraConfig.Default.position.z shouldBe 3f

  it should "look at origin (0, 0, 0)" in:
    CameraConfig.Default.lookAt.x shouldBe 0f
    CameraConfig.Default.lookAt.y shouldBe 0f
    CameraConfig.Default.lookAt.z shouldBe 0f

  it should "have up vector pointing in +Y direction" in:
    CameraConfig.Default.up.x shouldBe 0f
    CameraConfig.Default.up.y shouldBe 1f
    CameraConfig.Default.up.z shouldBe 0f

  // Edge case tests
  "CameraConfig edge cases" should "handle camera at origin" in:
    val config = CameraConfig(
      position = Vector3.Zero,
      lookAt = Vector3(0f, 0f, -1f),
      up = Vector3(0f, 1f, 0f)
    )
    config.position shouldBe Vector3.Zero

  it should "handle camera looking at itself (degenerate case)" in:
    // This is mathematically invalid but should not throw
    val config = CameraConfig(
      position = Vector3(1f, 1f, 1f),
      lookAt = Vector3(1f, 1f, 1f),
      up = Vector3(0f, 1f, 0f)
    )
    config.position shouldBe config.lookAt

  it should "handle non-normalized up vector" in:
    val config = CameraConfig(
      position = Vector3.Zero,
      lookAt = Vector3(0f, 0f, -1f),
      up = Vector3(0f, 100f, 0f)  // Not normalized
    )
    config.up.y shouldBe 100f

  it should "handle extreme coordinate values" in:
    val config = CameraConfig(
      position = Vector3(1000000f, -1000000f, 1000000f),
      lookAt = Vector3(-1000000f, 1000000f, -1000000f),
      up = Vector3(0f, 1f, 0f)
    )
    config.position.x shouldBe 1000000f
    config.lookAt.x shouldBe -1000000f

  it should "handle tilted up vector" in:
    val config = CameraConfig(
      position = Vector3.Zero,
      lookAt = Vector3(0f, 0f, -1f),
      up = Vector3(1f, 1f, 0f)  // 45-degree tilt
    )
    config.up.x shouldBe 1f
    config.up.y shouldBe 1f
