package menger.video

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class VideoModelsSuite extends AnyFlatSpec with Matchers:

  "VideoPlayback" should "default to animation-range looping with source fps" in:
    val playback = VideoPlayback()

    playback.timeMapping shouldBe VideoTimeMapping.AnimationRange
    playback.repeat shouldBe VideoRepeat.Loop
    playback.startOffset shouldBe 0.0
    playback.fpsOverride shouldBe None

  it should "accept positive fps override" in:
    VideoPlayback(fpsOverride = Some(24.0)).fpsOverride shouldBe Some(24.0)

  it should "reject non-positive fps override" in:
    an[IllegalArgumentException] should be thrownBy VideoPlayback(fpsOverride = Some(0.0))
    an[IllegalArgumentException] should be thrownBy VideoPlayback(fpsOverride = Some(-1.0))

  "VideoTexture" should "default to default playback" in:
    val texture = VideoTexture("clips/checker.mov")

    texture.path shouldBe "clips/checker.mov"
    texture.playback shouldBe VideoPlayback()

  it should "create a stable key from path and playback config" in:
    val first = VideoTexture(
      "clips/checker.mov",
      VideoPlayback(
        timeMapping = VideoTimeMapping.TSeconds,
        repeat = VideoRepeat.Freeze,
        startOffset = 0.5,
        fpsOverride = Some(24.0)
      )
    )
    val second = first.copy()

    second.textureKey shouldBe first.textureKey
    first.textureKey should include("clips/checker.mov")
    first.textureKey should include("TSeconds")
    first.textureKey should include("Freeze")
    first.textureKey should include("24.0")

  it should "reject blank paths" in:
    an[IllegalArgumentException] should be thrownBy VideoTexture("")
    an[IllegalArgumentException] should be thrownBy VideoTexture("   ")
