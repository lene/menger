package menger.common

case object Const:
  val epsilon: Float = 1e-5
  val defaultWindowWidth = 800
  val defaultWindowHeight = 600
  val defaultAntialiasSamples = 4
  val defaultScreenW = 1f
  val defaultEyeW = 2f
  val fpsLogIntervalMs = 1000

  // Angle conversion utilities
  def degreesToRadians(degrees: Float): Float = degrees * (math.Pi / 180.0).toFloat
  def radiansToDegrees(radians: Float): Float = radians * (180.0 / math.Pi).toFloat
