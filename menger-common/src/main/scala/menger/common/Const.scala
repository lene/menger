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

  // New constants extracted from MengerCLIOptions
  val maxLights = 8
  val defaultMaxInstances = 64

  // Input and interaction constants
  object Input:
    // Rotation and interaction
    val defaultRotateAngle: Float = 45f              // Default rotation increment in degrees
    val eyeScrollBase: Double = 64.0                // Base for exponential eyeW scroll calculation
    val eyeScrollOffset: Float = 1.0f               // Offset added to scroll result
    val fullRotationDegrees: Float = 360f              // Complete rotation in degrees
    
    // Sensitivities
    val rotation4DSensitivity: Float = 0.3f          // 4D rotation sensitivity factor
    val zoomSensitivity: Float = 0.1f                 // Zoom sensitivity
    
    // Orbit controls (merged from duplicate Input object below)
    val defaultZoomSensitivity = 0.3f
    val defaultPanSensitivity = 0.005f
    val defaultMinDistance = 0.1f
    val defaultMaxDistance = 20.0f
    val defaultMinElevation = -89.0f
    val defaultMaxElevation = 89.0f

  object Camera:
    // Perspective and projection
    val perspectiveScale: Float = 2.0f                // NDC range scale factor
    val pixelHalfUnit: Float = 1.0f                  // Unit for pixel calculations
  val maxInstancesLimit = 1024
  val maxPhotonsDefault = 100000
  val maxPhotonsLimit = 10000000
  val maxIterationsDefault = 10
  val maxIterationsLimit = 1000
  val maxCausticsRadius = 10.0f
  val rgbMaxValue = 255
  val rgbMaxValueFloat = 255f
  val defaultSpongeLevel = 1.0f

  // New constants extracted from OptiXEngine
  object Engine:
    val spongeLevelWarningThreshold = 3
    val cubeSpongeMaxLevel = 5
    val trianglesPerCube = 12
    val cubesPerSpongeLevel = 20
    val cubeScaleFactor = 2.0f
    val sphereScaleFactor = 2.0f
    val spongeScaleFactor = 2.0f

  // New constants extracted from OptiXRenderer
  object Renderer:
    val transformMatrixSize = 12  // 4x3 row-major
    val defaultMaxInstances = 64
    val streamCopyBufferSize = 8192
    val horizontalFov = 45f

  // New constants extracted from Main
  object Display:
    val colorBits = 8
    val depthBits = 16
    val stencilBits = 0

  

  // New constants extracted from Face4D
  object Geometry:
    val verticesPerFace = 4

