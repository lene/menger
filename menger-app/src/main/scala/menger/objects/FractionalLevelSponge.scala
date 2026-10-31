package menger.objects

import menger.common.TriangleMeshData
import menger.common.Vector


trait FractionalLevelSponge:
  def center: Vector[3]
  def scale: Float
  def level: Float

  /** Merge the next-level mesh with the hole caps of the current level into a single
   *  fractional-level mesh: nextLevel is fully opaque, the caps (`HoleCaps`) have
   *  alpha = 1 - fractionalPart, so the new holes fade in. Only the caps fade, not the whole
   *  current-level surface: that used to be laid 0.0003 outside the next level and caused
   *  speckle, double refraction on glass and shifted procedural colours (usability review
   *  2026-09, session 2, F35/F41). */
  protected def buildFractionalMesh(
    nextLevelMesh: TriangleMeshData,
    currentLevelMesh: TriangleMeshData
  ): TriangleMeshData =
    val alphaTransparent = 1.0f - (level - level.floor)
    TriangleMeshData.merge(Seq(
      TriangleMeshData.withAlpha(nextLevelMesh, 1.0f),
      TriangleMeshData.withAlpha(HoleCaps.of(currentLevelMesh), alphaTransparent)
    ))
