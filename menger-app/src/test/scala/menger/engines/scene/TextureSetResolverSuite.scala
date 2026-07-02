package menger.engines.scene

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TextureSetResolverSuite extends AnyFlatSpec with Matchers:

  private def resourceDir(name: String): Path =
    val candidates = List(
      Path.of(s"scripts/test-assets/texture-sets/$name"),
      Path.of(s"../scripts/test-assets/texture-sets/$name"),
      Path.of(s"menger-app/../scripts/test-assets/texture-sets/$name")
    )
    candidates.find(Files.isDirectory(_)).getOrElse(
      Path.of(s"scripts/test-assets/texture-sets/$name"))

  private def tempDir(body: Path => Unit): Unit =
    val dir = Files.createTempDirectory("pbr-test")
    try body(dir)
    finally
      dir.toFile.listFiles().foreach: f =>
        if f.isDirectory then f.listFiles().foreach(_.delete())
        f.delete()
      Files.deleteIfExists(dir)

  // ===== Poly Haven convention =====

  "TextureSetResolver" should "resolve tiny-pbr test set with Poly Haven convention" in:
    val result = TextureSetResolver.resolve(resourceDir("tiny-pbr"))
    result.isSuccess shouldBe true
    val set = result.get
    set.color.isDefined shouldBe true
    set.normal.isDefined shouldBe true
    set.roughness.isDefined shouldBe true
    set.metallic.isDefined shouldBe true
    set.ao.isDefined shouldBe true
    set.height.isDefined shouldBe false
    set.normalNeedsDXConversion shouldBe false

  it should "detect no convention for empty directory" in:
    tempDir: dir =>
      val result = TextureSetResolver.resolve(dir)
      result.isFailure shouldBe true
      result.failed.get.getMessage should include("No image files found")

  it should "reject non-directory path" in:
    val result = TextureSetResolver.resolve(Path.of("/nonexistent/path"))
    result.isFailure shouldBe true
    result.failed.get.getMessage should include("Not a directory")

  it should "set NormalDX flag for DirectX normals" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("mat_diff.png"), "x")
      Files.writeString(dir.resolve("mat_rough.png"), "x")
      Files.writeString(dir.resolve("mat_nor_dx.png"), "x")
      val result = TextureSetResolver.resolve(dir)
      result.isSuccess shouldBe true
      result.get.normalNeedsDXConversion shouldBe true

  it should "prefer NormalGL over NormalDX when both present" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("mat_diff.png"), "x")
      Files.writeString(dir.resolve("mat_rough.png"), "x")
      Files.writeString(dir.resolve("mat_nor_gl.png"), "x")
      Files.writeString(dir.resolve("mat_nor_dx.png"), "x")
      val result = TextureSetResolver.resolve(dir)
      result.isSuccess shouldBe true
      result.get.normal.isDefined shouldBe true
      result.get.normalNeedsDXConversion shouldBe false

  it should "detect height/displacement map" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("mat_diff.png"), "x")
      Files.writeString(dir.resolve("mat_rough.png"), "x")
      Files.writeString(dir.resolve("mat_disp.png"), "x")
      val result = TextureSetResolver.resolve(dir)
      result.isSuccess shouldBe true
      result.get.height.isDefined shouldBe true

  // ===== ambientCG convention =====

  it should "resolve ambientCG test set" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("mat_Color_4K.jpg"), "x")
      Files.writeString(dir.resolve("mat_NormalGL_4K.jpg"), "x")
      Files.writeString(dir.resolve("mat_Roughness_4K.jpg"), "x")
      Files.writeString(dir.resolve("mat_Metalness_4K.jpg"), "x")
      val result = TextureSetResolver.resolve(dir)
      result.isSuccess shouldBe true
      val set = result.get
      set.color.isDefined shouldBe true
      set.normal.isDefined shouldBe true
      set.roughness.isDefined shouldBe true
      set.metallic.isDefined shouldBe true

  it should "detect ambientCG DX normals" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("mat_Color_4K.jpg"), "x")
      Files.writeString(dir.resolve("mat_Roughness_4K.jpg"), "x")
      Files.writeString(dir.resolve("mat_NormalDX_4K.jpg"), "x")
      val result = TextureSetResolver.resolve(dir)
      result.isSuccess shouldBe true
      result.get.normalNeedsDXConversion shouldBe true

  it should "detect ambientCG AO and displacement" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("mat_Color_4K.jpg"), "x")
      Files.writeString(dir.resolve("mat_Roughness_4K.jpg"), "x")
      Files.writeString(dir.resolve("mat_AmbientOcclusion_4K.jpg"), "x")
      Files.writeString(dir.resolve("mat_Displacement_4K.jpg"), "x")
      val result = TextureSetResolver.resolve(dir)
      result.isSuccess shouldBe true
      val set = result.get
      set.ao.isDefined shouldBe true
      set.height.isDefined shouldBe true

  // ===== Convention detection edge cases =====

  it should "reject directory with only 2 known patterns" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("mat_diff.png"), "x")
      Files.writeString(dir.resolve("mat_rough.png"), "x")
      // Only 2 — needs at least 3 to trust convention
      val result = TextureSetResolver.resolve(dir)
      result.isFailure shouldBe true
      result.failed.get.getMessage should include("Could not detect PBR naming convention")

  it should "match case-insensitively" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("MAT_DIFF.PNG"), "x")
      Files.writeString(dir.resolve("MAT_ROUGH.PNG"), "x")
      Files.writeString(dir.resolve("MAT_NOR_GL.PNG"), "x")
      val result = TextureSetResolver.resolve(dir)
      result.isSuccess shouldBe true
      result.get.normal.isDefined shouldBe true

  it should "ignore non-image files" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("mat_diff.png"), "x")
      Files.writeString(dir.resolve("mat_rough.png"), "x")
      Files.writeString(dir.resolve("mat_nor_gl.png"), "x")
      Files.writeString(dir.resolve("readme.txt"), "docs")
      Files.writeString(dir.resolve("license.md"), "MIT")
      val result = TextureSetResolver.resolve(dir)
      result.isSuccess shouldBe true

  it should "detect Poly Haven over ambientCG when convention has more matches" in:
    tempDir: dir =>
      // 3 Poly Haven matches, 2 ambientCG matches
      Files.writeString(dir.resolve("mat_diff.png"), "x")
      Files.writeString(dir.resolve("mat_rough.png"), "x")
      Files.writeString(dir.resolve("mat_nor_gl.png"), "x")
      Files.writeString(dir.resolve("mat_Color_2K.jpg"), "x")
      Files.writeString(dir.resolve("mat_Roughness_2K.jpg"), "x")
      val result = TextureSetResolver.resolve(dir)
      result.isSuccess shouldBe true
      // Poly Haven wins (3 vs 2)
      result.get.normal.isDefined shouldBe true
