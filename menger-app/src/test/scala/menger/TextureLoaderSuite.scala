package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TextureLoaderSuite extends AnyFlatSpec with Matchers:

  "TextureLoader.createCheckerTexture" should "create texture with correct dimensions" in:
    val texture = TextureLoader.createCheckerTexture(64, 64)
    texture.width shouldBe 64
    texture.height shouldBe 64
    texture.data.length shouldBe 64 * 64 * 4

  it should "have fully opaque pixels" in:
    val texture = TextureLoader.createCheckerTexture(16, 16)
    for i <- texture.data.indices by 4 do
      (texture.data(i + 3) & 0xFF) shouldBe 255

  it should "create checkerboard pattern" in:
    val texture = TextureLoader.createCheckerTexture(16, 16, cellSize = 8)
    // Top-left corner should be one color, adjacent cell should be different
    val topLeft = texture.data(0) & 0xFF
    val topRight = texture.data(8 * 4) & 0xFF
    topLeft should not equal topRight

  "TextureLoader.createGradientTexture" should "create texture with correct dimensions" in:
    val texture = TextureLoader.createGradientTexture(32, 32)
    texture.width shouldBe 32
    texture.height shouldBe 32
    texture.data.length shouldBe 32 * 32 * 4

  it should "have gradient in red channel (left to right)" in:
    val texture = TextureLoader.createGradientTexture(256, 256)
    // Left edge should have low red values
    val leftRed = texture.data(0) & 0xFF
    // Right edge should have high red values
    val rightRed = texture.data((255 * 4)) & 0xFF
    leftRed shouldBe 0
    rightRed should be > 250

  it should "have gradient in green channel (top to bottom)" in:
    val texture = TextureLoader.createGradientTexture(256, 256)
    // Top edge should have low green values
    val topGreen = texture.data(1) & 0xFF
    // Bottom edge should have high green values
    val bottomGreen = texture.data((255 * 256 * 4) + 1) & 0xFF
    topGreen shouldBe 0
    bottomGreen should be > 250

  "TextureLoader.createSolidTexture" should "create uniform color" in:
    val texture = TextureLoader.createSolidTexture(8, 8, 128, 64, 32, 255)
    for i <- 0 until 8 * 8 do
      val idx = i * 4
      (texture.data(idx) & 0xFF) shouldBe 128
      (texture.data(idx + 1) & 0xFF) shouldBe 64
      (texture.data(idx + 2) & 0xFF) shouldBe 32
      (texture.data(idx + 3) & 0xFF) shouldBe 255

  "TextureLoader.load" should "fail for non-existent file" in:
    val result = TextureLoader.load("nonexistent.png", ".")
    result.isFailure shouldBe true

  it should "fail for invalid directory" in:
    val result = TextureLoader.load("test.png", "/nonexistent/directory")
    result.isFailure shouldBe true
