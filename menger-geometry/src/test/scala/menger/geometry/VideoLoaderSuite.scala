package menger.geometry

import java.nio.file.Paths

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class VideoLoaderSuite extends AnyFlatSpec with Matchers:

  "VideoLoader" should "read metadata from a deterministic RGBA MOV fixture" in:
    withLoader { loader =>
      loader.width shouldBe 2
      loader.height shouldBe 2
      loader.frameCount shouldBe 2
      loader.nativeFps shouldBe 2.0 +- Tolerance
      loader.durationSeconds should be > 0.0
    }

  it should "decode the requested frame as top-to-bottom RGBA bytes" in:
    withLoader { loader =>
      unsigned(loader.frameAt(0.0)) shouldBe FirstFrame
      unsigned(loader.frameAt(0.49)) shouldBe FirstFrame
      unsigned(loader.frameAt(0.5)) shouldBe SecondFrame
    }

  it should "clamp timestamps to the first and last decodable frames" in:
    withLoader { loader =>
      unsigned(loader.frameAt(-1.0)) shouldBe FirstFrame
      unsigned(loader.frameAt(100.0)) shouldBe SecondFrame
    }

  private def withLoader(test: VideoLoader => Unit): Unit =
    val loader = new VideoLoader(fixturePath)
    try test(loader)
    finally loader.close()

  private def fixturePath: String =
    val resource = getClass.getResource("/video/two-frame-rgba.mov")
    require(resource != null, "Missing video fixture resource")
    Paths.get(resource.toURI).toString

  private def unsigned(bytes: Array[Byte]): Seq[Int] = bytes.map(_ & ByteMask).toSeq

  private def rgba(red: Int, green: Int, blue: Int): Seq[Int] =
    Seq(red, green, blue, Opaque)

  private val Tolerance = 0.0001
  private val ByteMask = 0xff
  private val Opaque = 255
  private val Black = 0

  private val FirstFrame =
    rgba(Opaque, Black, Black) ++ rgba(Black, Opaque, Black) ++
      rgba(Black, Black, Opaque) ++ rgba(Opaque, Opaque, Opaque)

  private val SecondFrame =
    rgba(Black, Opaque, Opaque) ++ rgba(Opaque, Black, Opaque) ++
      rgba(Opaque, Opaque, Black) ++ rgba(Black, Black, Black)
