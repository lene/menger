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

  /**
   * Creates a 4x3 transform: T · Rz · Ry · Rx · S (ZYX Euler, applied right-to-left).
   * The object is scaled, rotated in-place at (tx, ty, tz), then translated.
   *
   * @param rx X-axis rotation in radians
   * @param ry Y-axis rotation in radians
   * @param rz Z-axis rotation in radians
   * @param scale Uniform scale factor
   * @param tx Translation on X axis
   * @param ty Translation on Y axis
   * @param tz Translation on Z axis
   * @return 12-element array representing 4x3 row-major transform matrix
   */
  def createEulerRotationScaleTranslation(
      rx: Float, ry: Float, rz: Float,
      scale: Float,
      tx: Float, ty: Float, tz: Float
  ): Array[Float] =
    val cx = math.cos(rx).toFloat; val sx = math.sin(rx).toFloat
    val cy = math.cos(ry).toFloat; val sy = math.sin(ry).toFloat
    val cz = math.cos(rz).toFloat; val sz = math.sin(rz).toFloat
    // R = Rz · Ry · Rx (columns of the combined rotation matrix)
    val r00 = cz * cy;             val r01 = cz * sy * sx - sz * cx; val r02 = cz * sy * cx + sz * sx
    val r10 = sz * cy;             val r11 = sz * sy * sx + cz * cx; val r12 = sz * sy * cx - cz * sx
    val r20 = -sy;                 val r21 = cy * sx;                 val r22 = cy * cx
    Array(
      scale * r00, scale * r01, scale * r02, tx,
      scale * r10, scale * r11, scale * r12, ty,
      scale * r20, scale * r21, scale * r22, tz
    )
