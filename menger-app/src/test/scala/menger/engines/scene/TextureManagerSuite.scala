package menger.engines.scene

import java.nio.file.Files
import java.nio.file.Paths
import java.util.concurrent.atomic.AtomicReference

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.TextureData
import menger.video.EnvMapVideo
import menger.video.VideoPlayback
import menger.video.VideoTexture
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TextureManagerSuite extends AnyFlatSpec with Matchers with MockFactory:

  "TextureManager" should "decode the initial video frame as texture data" in:
    val videoTexture = VideoTexture(FixtureName)
    val textureData = videoTextureDataOrCancel(
      TextureManager.loadInitialVideoTextureData(videoTexture, FixtureDir)
    )

    textureData.name shouldBe videoTexture.textureKey
    textureData.width shouldBe 2
    textureData.height shouldBe 2
    unsigned(textureData.data) shouldBe FirstFrame

  it should "decode the playback start offset as the initial video frame" in:
    val videoTexture = VideoTexture(FixtureName, VideoPlayback(startOffset = 0.5))
    val textureData = videoTextureDataOrCancel(
      TextureManager.loadInitialVideoTextureData(videoTexture, FixtureDir)
    )

    unsigned(textureData.data) shouldBe SecondFrame

  it should "decode the initial environment-map video frame as 2:1 texture data" in:
    val envMapVideo = EnvMapVideo(EnvMapFixtureName)
    val textureData = videoTextureDataOrCancel(
      TextureManager.loadInitialEnvMapVideoData(envMapVideo, FixtureDir)
    )

    textureData.name shouldBe envMapVideo.textureKey
    textureData.width shouldBe 4
    textureData.height shouldBe 2
    textureData.data.length shouldBe 4 * 2 * 4

  it should "reject environment-map videos that are not equirectangular 2:1" in:
    val failure = videoTextureFailureOrCancel(TextureManager.loadInitialEnvMapVideoData(
      EnvMapVideo(FixtureName),
      FixtureDir
    ))

    failure.getMessage should include("must be equirectangular 2:1")

  "ObjectSpec.imageTextureKey" should "use the video texture key when no image texture is set" in:
    val videoTexture = VideoTexture(FixtureName)
    val spec = ObjectSpec(objectType = "cube", videoTexture = Some(videoTexture))

    spec.imageTextureKey shouldBe Some(videoTexture.textureKey)

  "TextureManager video slot provider" should "reuse a supplied slot without uploading" in:
    val renderer = mock[OptiXRenderer]
    val videoTexture = VideoTexture(FixtureName)
    val spec = ObjectSpec(objectType = "cube", videoTexture = Some(videoTexture))
    val observed = AtomicReference(Option.empty[(VideoTexture, Int)])

    val indices = TextureManager.withVideoTextureSlotProvider(_ => Some(ReusableTextureIndex)) {
      TextureManager.withVideoTextureSlotObserver { (texture, textureIndex) =>
        observed.set(Some(texture -> textureIndex))
      } {
        TextureManager.loadTextures(List(spec), renderer, FixtureDir)
      }
    }

    indices shouldBe Map(videoTexture.textureKey -> ReusableTextureIndex)
    observed.get shouldBe Some(videoTexture -> ReusableTextureIndex)

  private def unsigned(bytes: Array[Byte]): Seq[Int] = bytes.map(_ & ByteMask).toSeq

  private def videoTextureDataOrCancel(result: Try[TextureData]): TextureData =
    result match
      case Success(textureData) => textureData
      case Failure(e: LinkageError) =>
        cancel(s"Video native library not available: ${e.getMessage}")
      case Failure(e) => fail(e)

  private def videoTextureFailureOrCancel(result: Try[TextureData]): Throwable =
    result match
      case Success(_) => fail("Expected video texture loading to fail")
      case Failure(e: LinkageError) =>
        cancel(s"Video native library not available: ${e.getMessage}")
      case Failure(e) => e

  private def rgba(red: Int, green: Int, blue: Int): Seq[Int] =
    Seq(red, green, blue, Opaque)

  private def FixtureDir =
    val repoRootPath = Paths.get("menger-geometry", "src", "test", "resources")
    if Files.exists(repoRootPath.resolve(FixtureName)) then repoRootPath.toString
    else Paths.get("..", "menger-geometry", "src", "test", "resources").toString

  private val FixtureName = "video/two-frame-rgba.mov"
  private val EnvMapFixtureName = "video/two-frame-equirect-rgba.mov"
  private val ReusableTextureIndex = 42
  private val ByteMask = 0xff
  private val Opaque = 255
  private val Black = 0

  private val FirstFrame =
    rgba(Opaque, Black, Black) ++ rgba(Black, Opaque, Black) ++
      rgba(Black, Black, Opaque) ++ rgba(Opaque, Opaque, Opaque)

  private val SecondFrame =
    rgba(Black, Opaque, Opaque) ++ rgba(Opaque, Black, Opaque) ++
      rgba(Opaque, Opaque, Black) ++ rgba(Black, Black, Black)
