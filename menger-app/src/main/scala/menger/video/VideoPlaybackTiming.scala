package menger.video

final case class AnimationTimeRange(start: Float, end: Float)

private def isUsableDuration(durationSeconds: Double): Boolean =
  java.lang.Double.isFinite(durationSeconds) && durationSeconds > 0.0

object VideoPlaybackTime:
  private val MinimumRangeSpan = 1.0e-9

  def resolve(
    playback: VideoPlayback,
    renderT: Float,
    animationRange: AnimationTimeRange,
    durationSeconds: Double
  ): Double =
    playback.startOffset + baseTime(
      playback.timeMapping,
      renderT,
      animationRange,
      durationSeconds
    )

  def sampleTime(
    playback: VideoPlayback,
    renderT: Float,
    animationRange: AnimationTimeRange,
    durationSeconds: Double
  ): Double =
    val rawTime = resolve(playback, renderT, animationRange, durationSeconds)
    val repeatedTime = VideoRepeatPolicy.resolve(rawTime, durationSeconds, playback.repeat)
    quantize(repeatedTime, playback.fpsOverride)

  def quantize(sampleTime: Double, fpsOverride: Option[Double]): Double =
    fpsOverride match
      case Some(fps) => Math.floor(sampleTime * fps) / fps
      case None      => sampleTime

  private def baseTime(
    mapping: VideoTimeMapping,
    renderT: Float,
    animationRange: AnimationTimeRange,
    durationSeconds: Double
  ): Double =
    mapping match
      case VideoTimeMapping.AnimationRange =>
        animationRangeTime(renderT, animationRange, durationSeconds)
      case VideoTimeMapping.TSeconds =>
        renderT.toDouble
      case VideoTimeMapping.TProgress =>
        progressTime(renderT, durationSeconds)

  private def animationRangeTime(
    renderT: Float,
    animationRange: AnimationTimeRange,
    durationSeconds: Double
  ): Double =
    if !isUsableDuration(durationSeconds) then 0.0
    else
      val span = (animationRange.end - animationRange.start).toDouble
      if Math.abs(span) <= MinimumRangeSpan then progressTime(renderT, durationSeconds)
      else ((renderT - animationRange.start).toDouble / span) * durationSeconds

  private def progressTime(renderT: Float, durationSeconds: Double): Double =
    if isUsableDuration(durationSeconds) then renderT.toDouble * durationSeconds
    else 0.0

object VideoRepeatPolicy:
  def resolve(rawTime: Double, durationSeconds: Double, repeat: VideoRepeat): Double =
    if !isUsableDuration(durationSeconds) then 0.0
    else
      repeat match
        case VideoRepeat.Loop =>
          positiveModulo(rawTime, durationSeconds)
        case VideoRepeat.Freeze =>
          rawTime.max(0.0).min(durationSeconds)
        case VideoRepeat.PingPong =>
          val phase = positiveModulo(rawTime, durationSeconds * 2.0)
          if phase <= durationSeconds then phase
          else durationSeconds * 2.0 - phase

  private def positiveModulo(value: Double, modulus: Double): Double =
    val remainder = value % modulus
    if remainder < 0.0 then remainder + modulus else remainder
