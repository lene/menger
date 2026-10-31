package menger.config

/** Configuration for t-parameter frame animation.
  *
  * @param startT Start value of the t parameter range
  * @param endT End value of the t parameter range
  * @param frames Total number of frames to render
  * @param savePattern Printf-style pattern for output filenames (must contain %), or empty for a
  *                    preview that saves nothing
  */
case class TAnimationConfig(
  startT: Float,
  endT: Float,
  frames: Int,
  savePattern: String
):
  require(frames > 0, "frames must be positive")
  // Empty is the preview's "no output": requiring % there made --preview fail on every
  // animated scene (found by the usability review 2026-09).
  require(
    savePattern.isEmpty || savePattern.contains("%"),
    "savePattern must contain % for frame numbering"
  )

  def tForFrame(frameIndex: Int): Float =
    if frames == 1 then startT
    else startT + frameIndex.toFloat * (endT - startT) / (frames - 1).toFloat
