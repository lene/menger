package menger.common

/**
 * OptiX rendering constants that mirror the C++ definitions in OptiXData.h.
 * These constants are used in tests and Scala code that interfaces with the OptiX renderer.
 *
 * IMPORTANT: Keep these values synchronized with optix-jni/src/main/native/include/OptiXData.h
 */
object OptiXConstants:
  // Default geometry values (matches RayTracingConstants in OptiXData.h)
  val defaultSphereRadius: Float = 1.5f
  val defaultCameraZDistance: Float = 3.0f
  val defaultFovDegrees: Float = 60.0f
  val defaultFloorPlaneY: Float = -2.0f

  // Material constants - Index of Refraction (matches MaterialConstants in OptiXData.h)
  val iorVacuum: Float = 1.0f
  val iorWater: Float = 1.33f
  val iorGlass: Float = 1.5f
  val iorDiamond: Float = 2.42f
