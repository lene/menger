package menger.objects

import scala.math.abs

import com.badlogic.gdx.math.Vector3

/**
 * Generates cube instance transforms for a Menger sponge.
 *
 * Unlike SpongeByVolume which merges all cubes into a single mesh,
 * this generator produces only the transform matrices for each cube,
 * enabling efficient GPU instancing via IAS.
 *
 * Algorithm: Recursive subdivision using the same pattern as SpongeByVolume,
 * but collecting transforms instead of merging geometry.
 *
 * Cube counts by level:
 * - Level 0: 1 cube
 * - Level 1: 20 cubes (3^3 - 7 removed = 27 - 7 = 20)
 * - Level 2: 400 cubes (20^2)
 * - Level 3: 8,000 cubes (20^3)
 * - Level 4: 160,000 cubes (20^4)
 * - Level 5: 3,200,000 cubes (20^5)
 *
 * @param center Center position of the sponge
 * @param size Overall size of the sponge
 * @param level Recursion level (integer only, fractional levels not supported)
 */
case class CubeSpongeGenerator(
  center: Vector3 = Vector3.Zero,
  size: Float = 1.0f,
  level: Int = 1
):
  require(level >= 0, s"Level must be non-negative, got $level")
  require(size > 0, s"Size must be positive, got $size")

  /**
   * Generate all cube transforms for this sponge.
   * Returns a sequence of (position, scale) tuples.
   */
  def generateTransforms: Seq[(Vector3, Float)] =
    if level == 0 then
      // Base case: single cube
      Seq((center, size))
    else
      // Recursive case: generate 20 sub-sponges per iteration
      generateRecursive(center, size, level)

  private def generateRecursive(center: Vector3, size: Float, level: Int): Seq[(Vector3, Float)] =
    if level == 0 then
      Seq((center, size))
    else
      val shift = size / 3.0f
      val subSize = size / 3.0f

      // Generate 20 positions (3^3 - 7 removed = 20)
      // Remove center (0,0,0) and 6 face centers (±1,0,0), (0,±1,0), (0,0,±1)
      val positions = for
        xx <- -1 to 1
        yy <- -1 to 1
        zz <- -1 to 1
        if abs(xx) + abs(yy) + abs(zz) > 1  // Keep only edge and corner cubes
      yield Vector3(center.x + xx * shift, center.y + yy * shift, center.z + zz * shift)

      // Recursively subdivide each position
      positions.flatMap { pos =>
        generateRecursive(pos, subSize, level - 1)
      }

  /**
   * Count total number of cubes that will be generated.
   * Useful for validation and memory estimation.
   */
  def cubeCount: Long =
    if level == 0 then 1L
    else Math.pow(20, level).toLong

  /**
   * Estimate GPU memory usage for transforms only (not geometry).
   * Each transform is 4x3 floats = 48 bytes
   */
  def estimateTransformMemoryBytes: Long =
    cubeCount * 48L  // 4x3 transform matrix = 12 floats = 48 bytes

  override def toString: String =
    s"CubeSpongeGenerator(level=$level, cubes=$cubeCount, center=$center, size=$size)"
