package menger.engines.scene

import java.nio.file.Files
import java.nio.file.Paths

import menger.ObjectSpec
import menger.video.VideoPlayback
import menger.video.VideoTexture
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TextureManagerSuite extends AnyFlatSpec with Matchers:

  "TextureManager" should "decode the initial video frame as texture data" in:
    val videoTexture = VideoTexture(FixtureName)
    val textureData = TextureManager.loadInitialVideoTextureData(videoTexture, FixtureDir).get

    textureData.name shouldBe videoTexture.textureKey
    textureData.width shouldBe 2
    textureData.height shouldBe 2
    unsigned(textureData.data) shouldBe FirstFrame

  it should "decode the playback start offset as the initial video frame" in:
    val videoTexture = VideoTexture(FixtureName, VideoPlayback(startOffset = 0.5))
    val textureData = TextureManager.loadInitialVideoTextureData(videoTexture, FixtureDir).get

    unsigned(textureData.data) shouldBe SecondFrame

  "ObjectSpec.imageTextureKey" should "use the video texture key when no image texture is set" in:
    val videoTexture = VideoTexture(FixtureName)
    val spec = ObjectSpec(objectType = "cube", videoTexture = Some(videoTexture))

    spec.imageTextureKey shouldBe Some(videoTexture.textureKey)

  private def unsigned(bytes: Array[Byte]): Seq[Int] = bytes.map(_ & ByteMask).toSeq

  private def rgba(red: Int, green: Int, blue: Int): Seq[Int] =
    Seq(red, green, blue, Opaque)

  private def FixtureDir =
    val repoRootPath = Paths.get("menger-geometry", "src", "test", "resources")
    if Files.exists(repoRootPath.resolve(FixtureName)) then repoRootPath.toString
    else Paths.get("..", "menger-geometry", "src", "test", "resources").toString

  private val FixtureName = "video/two-frame-rgba.mov"
  private val ByteMask = 0xff
  private val Opaque = 255
  private val Black = 0

  private val FirstFrame =
    rgba(Opaque, Black, Black) ++ rgba(Black, Opaque, Black) ++
      rgba(Black, Black, Opaque) ++ rgba(Opaque, Opaque, Opaque)

  private val SecondFrame =
    rgba(Black, Opaque, Opaque) ++ rgba(Opaque, Black, Opaque) ++
      rgba(Opaque, Opaque, Black) ++ rgba(Black, Black, Black)
