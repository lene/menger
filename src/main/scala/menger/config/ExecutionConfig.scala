package menger.config

/**
 * Execution configuration for runtime behavior.
 *
 * @param fpsLogIntervalMs interval in milliseconds between FPS log messages (0 = disabled)
 * @param timeout automatic exit timeout in seconds (0 = no timeout)
 * @param saveName optional screenshot filename (with %d for frame number in animations)
 * @param enableStats enable ray tracing statistics collection and logging
 * @param maxInstances maximum number of object instances allowed in multi-object scenes
 * @param textureDir base directory for loading texture files
 */
case class ExecutionConfig(
  fpsLogIntervalMs: Int = 5000,
  timeout: Float = 0f,
  saveName: Option[String] = None,
  enableStats: Boolean = false,
  maxInstances: Int = 64,
  textureDir: String = "."
)

object ExecutionConfig:
  /**
   * Default configuration: 5s FPS logging, no timeout, no screenshots, no stats
   */
  val Default: ExecutionConfig = ExecutionConfig()

  /**
   * Configuration for testing: short timeout, no logging
   */
  val Testing: ExecutionConfig = ExecutionConfig(
    fpsLogIntervalMs = 0,
    timeout = 0.1f,
    saveName = None,
    enableStats = false,
    maxInstances = 64,
    textureDir = "."
  )

  /**
   * Configuration for benchmarking: statistics enabled, no logging
   */
  val Benchmarking: ExecutionConfig = ExecutionConfig(
    fpsLogIntervalMs = 0,
    timeout = 0f,
    saveName = None,
    enableStats = true,
    maxInstances = 64,
    textureDir = "."
  )
