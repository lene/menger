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

  it should "read metadata from a deterministic equirectangular MOV fixture" in:
    withLoader(equirectFixturePath) { loader =>
      loader.width shouldBe 4
      loader.height shouldBe 2
      loader.frameCount shouldBe 2
      loader.nativeFps shouldBe 2.0 +- Tolerance
      loader.durationSeconds should be > 0.0
      loader.frameAt(0.0).toSeq should not be loader.frameAt(0.5).toSeq
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

  it should "prefetch frames without changing decoded output" in:
    withLoader { loader =>
      loader.prefetch(0.0, 2)

      unsigned(loader.frameAt(0.0)) shouldBe FirstFrame
      unsigned(loader.frameAt(0.5)) shouldBe SecondFrame
    }

  it should "close idempotently and reject later access" in:
    withLoader { loader =>
      loader.close()
      loader.close()

      an[IllegalArgumentException] shouldBe thrownBy(loader.frameAt(0.0))
    }

  private def withLoader(test: VideoLoader => Unit): Unit =
    withLoader(fixturePath)(test)

  private def withLoader(path: String)(test: VideoLoader => Unit): Unit =
    try
      val loader = new VideoLoader(path)
      try test(loader)
      finally loader.close()
    catch
      case e: LinkageError =>
        cancel(s"Video native library not available: ${e.getMessage}")

  private def fixturePath: String =
    resourcePath("/video/two-frame-rgba.mov")

  private def equirectFixturePath: String =
    resourcePath("/video/two-frame-equirect-rgba.mov")

  private def resourcePath(name: String): String =
    val resource = getClass.getResource(name)
    require(resource != null, s"Missing video fixture resource: $name")
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
