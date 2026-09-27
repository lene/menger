package menger

import java.nio.file.Files

import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AssetPathsSuite extends AnyFlatSpec with Matchers with BeforeAndAfterAll:

  private val baseDir = Files.createTempDirectory("asset-paths-suite")
  private val outsideDir = Files.createTempDirectory("asset-paths-suite-outside")

  override def afterAll(): Unit =
    Files.deleteIfExists(baseDir.resolve("inside.png"))
    Files.deleteIfExists(baseDir)
    Files.deleteIfExists(outsideDir.resolve("secret.png"))
    Files.deleteIfExists(outsideDir)

  "AssetPaths.resolve" should "accept a relative path inside baseDir" in:
    val result = AssetPaths.resolve(baseDir.toString, "texture.png")
    result shouldBe Right(baseDir.resolve("texture.png"))

  it should "accept a nested relative path inside baseDir" in:
    val result = AssetPaths.resolve(baseDir.toString, "set/color.png")
    result shouldBe Right(baseDir.resolve("set/color.png"))

  it should "reject an absolute path" in:
    val result = AssetPaths.resolve(baseDir.toString, "/etc/passwd")
    result.isLeft shouldBe true
    result.left.toOption.get should include("absolute")

  it should "reject a .. escape" in:
    val result = AssetPaths.resolve(baseDir.toString, "../outside.png")
    result.isLeft shouldBe true
    result.left.toOption.get should include("escapes")

  it should "reject a .. escape buried inside a deeper relative path" in:
    val result = AssetPaths.resolve(baseDir.toString, "set/../../outside.png")
    result.isLeft shouldBe true
    result.left.toOption.get should include("escapes")

  it should "reject a symlink that points outside baseDir" in:
    val target = outsideDir.resolve("secret.png")
    Files.write(target, Array[Byte](1, 2, 3))
    val link = baseDir.resolve("inside.png")
    Files.deleteIfExists(link)
    Files.createSymbolicLink(link, target)
    try
      val result = AssetPaths.resolve(baseDir.toString, "inside.png")
      result.isLeft shouldBe true
      result.left.toOption.get should include("symlink")
    finally
      Files.deleteIfExists(link)

  it should "return Right for a missing file inside baseDir (let the caller report not-found)" in:
    val result = AssetPaths.resolve(baseDir.toString, "does-not-exist.png")
    result shouldBe Right(baseDir.resolve("does-not-exist.png"))

  "AssetPaths.resolveOrThrow" should "throw AssetPathException for an absolute path" in:
    an[AssetPaths.AssetPathException] should be thrownBy:
      AssetPaths.resolveOrThrow(baseDir.toString, "/etc/passwd")

  it should "return the resolved path when valid" in:
    AssetPaths.resolveOrThrow(baseDir.toString, "texture.png") shouldBe baseDir.resolve("texture.png")
