package menger.video

enum VideoTimeMapping:
  case AnimationRange
  case TSeconds
  case TProgress

enum VideoRepeat:
  case Loop
  case Freeze
  case PingPong

case class VideoPlayback(
  timeMapping: VideoTimeMapping = VideoTimeMapping.AnimationRange,
  repeat: VideoRepeat = VideoRepeat.Loop,
  startOffset: Double = 0.0,
  fpsOverride: Option[Double] = None
):
  require(fpsOverride.forall(_ > 0.0), "fpsOverride must be positive when set")

  def textureKey(prefix: String, path: String): String =
    val fps = fpsOverride.map(_.toString).getOrElse("source")
    s"$prefix:$path|time=$timeMapping|repeat=$repeat|offset=$startOffset|fps=$fps"

case class VideoTexture(
  path: String,
  playback: VideoPlayback = VideoPlayback()
):
  require(path.trim.nonEmpty, "Video texture path cannot be empty")

  def textureKey: String = playback.textureKey("video", path)

case class EnvMapVideo(
  path: String,
  playback: VideoPlayback = VideoPlayback()
):
  require(path.trim.nonEmpty, "Environment-map video path cannot be empty")

  def textureKey: String = playback.textureKey("env-video", path)
