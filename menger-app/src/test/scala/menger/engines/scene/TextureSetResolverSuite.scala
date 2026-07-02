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

  "TextureSetResolver" should "resolve tiny-pbr test set with Poly Haven convention" in:
    val result = TextureSetResolver.resolve(resourceDir("tiny-pbr"))
    result.isSuccess shouldBe true
    val set = result.get
    set.color.isDefined shouldBe true
    set.normal.isDefined shouldBe true
    set.roughness.isDefined shouldBe true
    set.metallic.isDefined shouldBe true
    set.ao.isDefined shouldBe true
    // height map not in test set
    set.height.isDefined shouldBe false

  it should "detect no convention for empty directory" in:
    val tmpDir = Files.createTempDirectory("empty-pbr")
    try
      val result = TextureSetResolver.resolve(tmpDir)
      result.isFailure shouldBe true
      result.failed.get.getMessage should include("No image files found")
    finally
      Files.deleteIfExists(tmpDir)

  it should "reject non-directory path" in:
    val result = TextureSetResolver.resolve(Path.of("/nonexistent/path"))
    result.isFailure shouldBe true
    result.failed.get.getMessage should include("Not a directory")

  it should "set NormalDX flag for DirectX normals" in:
    // Create a temporary dir with a DX normal file
    val tmpDir = Files.createTempDirectory("dx-test")
    try
      Files.writeString(tmpDir.resolve("mat_diff.png"), "fake")
      Files.writeString(tmpDir.resolve("mat_rough.png"), "fake")
      Files.writeString(tmpDir.resolve("mat_nor_dx.png"), "fake")
      val result = TextureSetResolver.resolve(tmpDir)
      result.isSuccess shouldBe true
      result.get.normalNeedsDXConversion shouldBe true
    finally
      tmpDir.toFile.listFiles().foreach(_.delete())
      Files.deleteIfExists(tmpDir)
