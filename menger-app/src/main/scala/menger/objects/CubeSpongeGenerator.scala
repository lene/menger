package menger.objects

import scala.math.abs

import com.badlogic.gdx.math.Vector3

/**
 * Generates cube instance transforms for a Menger sponge.
 *
 * Unlike SpongeByVolume which merges all cubes into a single mesh,
 * this generator produces transform matrices and alpha values for each
 * cube, enabling efficient GPU instancing via IAS.
 *
 * Algorithm: Recursive subdivision using the same pattern as SpongeByVolume,
 * but collecting transforms instead of merging geometry.
 *
 * Cube counts by integer level:
 * - Level 0: 1 cube
 * - Level 1: 20 cubes (3^3 - 7 removed = 27 - 7 = 20)
 * - Level 2: 400 cubes (20^2)
 * - Level 3: 8,000 cubes (20^3)
 * - Level 4: 160,000 cubes (20^4)
 * - Level 5: 3,200,000 cubes (20^5)
 *
 * Fractional levels (e.g. 1.5) animate the transition between integer levels
 * using alpha transparency:
 *
 * At level n+frac, the scene contains:
 *  - The full level-(n+1) sponge as solid cubes (alpha = 1.0)
 *  - 7 "tunnel-fill" ghost cubes per level-n cube (alpha = 1 - frac)
 *
 * The ghost cubes occupy the 7 sub-positions per level-n cube that are NOT
 * part of the Menger pattern (the center and 6 face-centers). At frac=0 they
 * are fully opaque, filling the tunnels so the structure looks like level-n.
 * As frac→1 they fade to transparent, revealing the level-(n+1) tunnel pattern.
 *
 * Instance counts at fractional levels:
 * - Level 0+frac: 20 solid + 7 ghost = 27
 * - Level 1+frac: 400 solid + 140 ghost = 540
 * - Level 2+frac: 8000 solid + 2800 ghost = 10800 (needs --max-instances 11000)
 *
 * @param center Center position of the sponge
 * @param size Overall size of the sponge
 * @param level Recursion level; fractional values animate the transition to the next level
 */
case class CubeSpongeGenerator(
  center: Vector3 = Vector3.Zero,
  size: Float = 1.0f,
  level: Float = 1.0f
):
  require(level >= 0, s"Level must be non-negative, got $level")
  require(size > 0, s"Size must be positive, got $size")

  private val intLevel: Int   = level.toInt
  private val fracPart: Float = level - intLevel

  /**
   * Generate all cube transforms for this sponge.
   * Returns a sequence of (position, scale, alpha) tuples.
   *
   * For integer levels alpha is always 1.0. For fractional levels the result
   * contains level-(intLevel+1) solid cubes (alpha=1.0) plus ghost tunnel-fill
   * cubes (alpha = 1 - fracPart) that fade out to reveal the next level's holes.
   */
  def generateTransforms: Seq[(Vector3, Float, Float)] =
    if fracPart < 1e-4f then
      generateInteger(center, size, intLevel).map { case (p, s) => (p, s, 1.0f) }
    else
      val solidCubes = generateInteger(center, size, intLevel + 1)
        .map { case (p, s) => (p, s, 1.0f) }
      val baseCubes  = generateInteger(center, size, intLevel)
      val ghostAlpha = 1f - fracPart
      val ghostCubes = baseCubes.flatMap { case (basePos, baseSize) =>
        generateGhostSubcubes(basePos, baseSize, ghostAlpha)
      }
      solidCubes ++ ghostCubes

  // The 7 positions inside a cube that are NOT in the Menger sponge pattern
  // (center + 6 face-centers): these fill the tunnels during the transition.
  private def generateGhostSubcubes(c: Vector3, s: Float, alpha: Float): Seq[(Vector3, Float, Float)] =
    val shift   = s / 3.0f
    val subSize = s / 3.0f
    for
      xx <- -1 to 1
      yy <- -1 to 1
      zz <- -1 to 1
      if abs(xx) + abs(yy) + abs(zz) <= 1   // center (0,0,0) + 6 face-centers
    yield (Vector3(c.x + xx * shift, c.y + yy * shift, c.z + zz * shift), subSize, alpha)

  private def generateInteger(c: Vector3, s: Float, n: Int): Seq[(Vector3, Float)] =
    if n == 0 then
      Seq((c, s))
    else
      val shift   = s / 3.0f
      val subSize = s / 3.0f
      val positions = for
        xx <- -1 to 1
        yy <- -1 to 1
        zz <- -1 to 1
        if abs(xx) + abs(yy) + abs(zz) > 1
      yield Vector3(c.x + xx * shift, c.y + yy * shift, c.z + zz * shift)
      positions.flatMap { pos => generateInteger(pos, subSize, n - 1) }

  /**
   * Count total number of cube instances that will be generated.
   * Useful for validation and memory estimation.
   */
  def cubeCount: Long =
    if fracPart < 1e-4f then
      if intLevel == 0 then 1L else math.pow(20, intLevel).toLong
    else
      val solidCount = math.pow(20, intLevel + 1).toLong
      val baseCount  = if intLevel == 0 then 1L else math.pow(20, intLevel).toLong
      solidCount + 7L * baseCount

  /**
   * Estimate GPU memory usage for transforms only (not geometry).
   * Each transform is 4x3 floats = 48 bytes
   */
  def estimateTransformMemoryBytes: Long =
    cubeCount * 48L  // 4x3 transform matrix = 12 floats = 48 bytes

  override def toString: String =
    s"CubeSpongeGenerator(level=$level, cubes=$cubeCount, center=$center, size=$size)"
