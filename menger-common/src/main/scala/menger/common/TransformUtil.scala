package menger.common

/**
 * Utilities for creating transformation matrices for OptiX geometry instances.
 *
 * Transform matrices use a 4x3 row-major format:
 * [row0: sx, 0, 0, tx], [row1: 0, sy, 0, ty], [row2: 0, 0, sz, tz]
 * where s = scale, t = translation
 */
object TransformUtil:

  /**
   * Creates a 4x3 transformation matrix with uniform scale and translation.
   *
   * @param scale Uniform scale factor applied to all axes
   * @param x Translation on X axis
   * @param y Translation on Y axis
   * @param z Translation on Z axis
   * @return 12-element array representing 4x3 row-major transform matrix
   */
  def createScaleTranslation(scale: Float, x: Float, y: Float, z: Float): Array[Float] =
    Array(
      scale, 0f, 0f, x,
      0f, scale, 0f, y,
      0f, 0f, scale, z
    )

  /**
   * Creates an identity transform (no scale, no translation).
   *
   * @return 12-element array representing identity transform
   */
  def identity(): Array[Float] =
    createScaleTranslation(1f, 0f, 0f, 0f)

  /**
   * Creates a translation-only transform (no scaling).
   *
   * @param x Translation on X axis
   * @param y Translation on Y axis
   * @param z Translation on Z axis
   * @return 12-element array representing translation-only transform
   */
  def translation(x: Float, y: Float, z: Float): Array[Float] =
    createScaleTranslation(1f, x, y, z)

  /**
   * Creates a uniform scale-only transform (no translation).
   *
   * @param scale Uniform scale factor applied to all axes
   * @return 12-element array representing scale-only transform
   */
  def uniformScale(scale: Float): Array[Float] =
    createScaleTranslation(scale, 0f, 0f, 0f)

  /**
   * Creates a 4x3 transform that applies uniform scale, Y-axis rotation, and translation,
   * where the translation is the LOCAL (pre-rotation) position of the object. Both the
   * object's orientation and its world position are rotated by yAngle.
   *
   * Useful for rotating an entire scene's-worth of instances around the world Y axis.
   *
   * @param yAngle Y-axis rotation in radians
   * @param scale  Uniform scale factor
   * @param lx     Local (pre-rotation) X position
   * @param ly     Local Y position (unchanged by Y-axis rotation)
   * @param lz     Local (pre-rotation) Z position
   * @return 12-element array representing 4x3 row-major transform matrix
   */
  def createYRotationScaleTranslation(yAngle: Float, scale: Float, lx: Float, ly: Float, lz: Float): Array[Float] =
    val cosA = math.cos(yAngle).toFloat
    val sinA = math.sin(yAngle).toFloat
    Array(
       scale * cosA, 0f,  scale * sinA,  cosA * lx + sinA * lz,
       0f,           scale, 0f,          ly,
      -scale * sinA, 0f,  scale * cosA, -sinA * lx + cosA * lz
    )
