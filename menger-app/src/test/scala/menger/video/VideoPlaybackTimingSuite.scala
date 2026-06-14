package menger.video

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class VideoPlaybackTimingSuite extends AnyFlatSpec with Matchers:

  "VideoPlaybackTime" should "map animation-range t over the source duration" in:
    val playback = VideoPlayback(timeMapping = VideoTimeMapping.AnimationRange)
    val range = AnimationTimeRange(start = 10f, end = 20f)

    VideoPlaybackTime.resolve(playback, 10f, range, SourceDuration) shouldBe 0.0
    VideoPlaybackTime.resolve(playback, 15f, range, SourceDuration) shouldBe 2.0
    VideoPlaybackTime.resolve(playback, 20f, range, SourceDuration) shouldBe 4.0

  it should "fall back to progress mapping when the animation range is unusable" in:
    val playback = VideoPlayback(timeMapping = VideoTimeMapping.AnimationRange)
    val range = AnimationTimeRange(start = 1f, end = 1f)

    VideoPlaybackTime.resolve(playback, 0.5f, range, SourceDuration) shouldBe 2.0

  it should "map TSeconds directly from render t" in:
    val playback = VideoPlayback(
      timeMapping = VideoTimeMapping.TSeconds,
      startOffset = StartOffset
    )

    VideoPlaybackTime.resolve(
      playback,
      2f,
      AnimationTimeRange(start = 0f, end = 1f),
      SourceDuration
    ) shouldBe 3.25

  it should "map TProgress from render t multiplied by duration" in:
    val playback = VideoPlayback(
      timeMapping = VideoTimeMapping.TProgress,
      startOffset = StartOffset
    )

    VideoPlaybackTime.resolve(
      playback,
      0.5f,
      AnimationTimeRange(start = 0f, end = 1f),
      SourceDuration
    ) shouldBe 3.25

  it should "quantize sampled time when fpsOverride is set" in:
    VideoPlaybackTime.quantize(0.49, Some(2.0)) shouldBe 0.0
    VideoPlaybackTime.quantize(0.50, Some(2.0)) shouldBe 0.5
    VideoPlaybackTime.quantize(0.99, Some(2.0)) shouldBe 0.5

  it should "leave sampled time unquantized when fpsOverride is absent" in:
    VideoPlaybackTime.quantize(0.49, None) shouldBe 0.49

  it should "quantize after repeat policy is applied" in:
    val playback = VideoPlayback(
      timeMapping = VideoTimeMapping.TSeconds,
      repeat = VideoRepeat.Loop,
      fpsOverride = Some(2.0)
    )

    VideoPlaybackTime.sampleTime(
      playback,
      -0.1f,
      AnimationTimeRange(start = 0f, end = 1f),
      durationSeconds = 1.0
    ) shouldBe 0.5

  "VideoRepeatPolicy" should "loop duration boundaries and negative time" in:
    VideoRepeatPolicy.resolve(SourceDuration, SourceDuration, VideoRepeat.Loop) shouldBe 0.0
    VideoRepeatPolicy.resolve(6.5, SourceDuration, VideoRepeat.Loop) shouldBe 2.5
    VideoRepeatPolicy.resolve(-0.25, SourceDuration, VideoRepeat.Loop) shouldBe 3.75

  it should "freeze before the first frame and after the last frame" in:
    VideoRepeatPolicy.resolve(-1.0, SourceDuration, VideoRepeat.Freeze) shouldBe 0.0
    VideoRepeatPolicy.resolve(5.0, SourceDuration, VideoRepeat.Freeze) shouldBe SourceDuration

  it should "ping-pong repeatedly across the duration" in:
    VideoRepeatPolicy.resolve(0.5, SourceDuration, VideoRepeat.PingPong) shouldBe 0.5
    VideoRepeatPolicy.resolve(SourceDuration, SourceDuration, VideoRepeat.PingPong) shouldBe
      SourceDuration
    VideoRepeatPolicy.resolve(5.0, SourceDuration, VideoRepeat.PingPong) shouldBe 3.0
    VideoRepeatPolicy.resolve(8.5, SourceDuration, VideoRepeat.PingPong) shouldBe 0.5
    VideoRepeatPolicy.resolve(-0.5, SourceDuration, VideoRepeat.PingPong) shouldBe 0.5

  private val SourceDuration = 4.0
  private val StartOffset = 1.25
