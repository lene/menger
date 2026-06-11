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

case class VideoTexture(
  path: String,
  playback: VideoPlayback = VideoPlayback()
):
  require(path.trim.nonEmpty, "Video texture path cannot be empty")

  def textureKey: String =
    val fps = playback.fpsOverride.map(_.toString).getOrElse("source")
    s"video:$path|time=${playback.timeMapping}|repeat=${playback.repeat}|" +
      s"offset=${playback.startOffset}|fps=$fps"
