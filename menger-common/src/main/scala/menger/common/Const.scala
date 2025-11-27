package menger.common

/**
 * Application-wide constants.
 *
 * OptiX rendering constants mirror C++ definitions in OptiXData.h.
 * IMPORTANT: Keep OptiX values synchronized with optix-jni/src/main/native/include/OptiXData.h
 */
case object Const:
  val epsilon: Float = 1e-5
  val defaultWindowWidth = 800
  val defaultWindowHeight = 600
  val defaultAntialiasSamples = 4
  val defaultScreenW = 1f
  val defaultEyeW = 2f
  val fpsLogIntervalMs = 1000

  // Angle conversion constants
  val degToRad: Float = (math.Pi / 180.0).toFloat
  val radToDeg: Float = (180.0 / math.Pi).toFloat

  // Angle conversion utilities
  def degreesToRadians(degrees: Float): Float = degrees * degToRad
  def radiansToDegrees(radians: Float): Float = radians * radToDeg

  // OptiX rendering constants (matches RayTracingConstants in OptiXData.h)
  val defaultSphereRadius: Float = 1.5f
  val defaultCameraZDistance: Float = 3.0f
  val defaultFovDegrees: Float = 60.0f
  val defaultFloorPlaneY: Float = -2.0f

  // Material constants - Index of Refraction (matches MaterialConstants in OptiXData.h)
  val iorVacuum: Float = 1.0f
  val iorWater: Float = 1.33f
  val iorGlass: Float = 1.5f
  val iorDiamond: Float = 2.42f
