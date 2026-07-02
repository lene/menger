package menger.engines.scene

import java.nio.file.{Files, Path}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TextureSetMetadataSuite extends AnyFlatSpec with Matchers:

  private def tempDir(body: Path => Unit): Unit =
    val dir = Files.createTempDirectory("meta-test")
    try body(dir)
    finally
      dir.toFile.listFiles().foreach(_.delete())
      Files.deleteIfExists(dir)

  "TextureSetMetadata" should "return empty metadata when sidecar is missing" in:
    tempDir: dir =>
      val result = TextureSetMetadata.load(dir)
      result.isSuccess shouldBe true
      val meta = result.get
      meta.ior shouldBe None
      meta.uvScale shouldBe None

  it should "parse valid JSON sidecar" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("menger-textureset.json"),
        """{"ior": 1.45, "uvScale": 2.0}""")
      val result = TextureSetMetadata.load(dir)
      result.isSuccess shouldBe true
      val meta = result.get
      meta.ior shouldBe Some(1.45f)
      meta.uvScale shouldBe Some(2.0f)

  it should "handle JSON with only ior" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("menger-textureset.json"),
        """{"ior": 2.42}""")
      val result = TextureSetMetadata.load(dir)
      result.isSuccess shouldBe true
      val meta = result.get
      meta.ior shouldBe Some(2.42f)
      meta.uvScale shouldBe None

  it should "handle JSON with only uvScale" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("menger-textureset.json"),
        """{"uvScale": 4.0}""")
      val result = TextureSetMetadata.load(dir)
      result.isSuccess shouldBe true
      val meta = result.get
      meta.ior shouldBe None
      meta.uvScale shouldBe Some(4.0f)

  it should "reject ior < 1.0" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("menger-textureset.json"),
        """{"ior": 0.5}""")
      val result = TextureSetMetadata.load(dir)
      result.isFailure shouldBe true
      result.failed.get.getMessage should include("IOR must be >= 1.0")

  it should "reject uvScale <= 0" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("menger-textureset.json"),
        """{"uvScale": -1.0}""")
      val result = TextureSetMetadata.load(dir)
      result.isFailure shouldBe true
      result.failed.get.getMessage should include("uvScale must be > 0")

  it should "reject malformed JSON" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("menger-textureset.json"),
        """{not json}""")
      val result = TextureSetMetadata.load(dir)
      result.isFailure shouldBe true

  it should "ignore extra fields" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("menger-textureset.json"),
        """{"ior": 1.5, "uvScale": 2.0, "unknown": "ignored"}""")
      val result = TextureSetMetadata.load(dir)
      result.isSuccess shouldBe true
      val meta = result.get
      meta.ior shouldBe Some(1.5f)
      meta.uvScale shouldBe Some(2.0f)

  it should "handle empty JSON object" in:
    tempDir: dir =>
      Files.writeString(dir.resolve("menger-textureset.json"),
        """{}""")
      val result = TextureSetMetadata.load(dir)
      result.isSuccess shouldBe true
      val meta = result.get
      meta.ior shouldBe None
      meta.uvScale shouldBe None
