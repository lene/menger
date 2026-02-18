package menger.objects

import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData


trait FractionalLevelSponge extends FractionalLevelObject:
  def center: Vector3
  def scale: Float
  def primitiveType: Int

  /** Merge next-level and current-level meshes into a single fractional-level mesh.
   *  Expands currentLevelMesh outward along normals to prevent z-fighting, then
   *  blends the two meshes using per-vertex alpha based on the fractional part of level. */
  protected def buildFractionalMesh(
    nextLevelMesh: TriangleMeshData,
    currentLevelMesh: TriangleMeshData
  ): TriangleMeshData =
    val alphaTransparent = 1.0f - (level - level.floor)
    val expanded = TriangleMeshData.expandAlongNormals(currentLevelMesh, FractionalLevelSponge.SkinNormalOffset)
    TriangleMeshData.merge(Seq(
      TriangleMeshData.withAlpha(nextLevelMesh, 1.0f),
      TriangleMeshData.withAlpha(expanded, alphaTransparent)
    ))

  
  protected def createInstance(
    center: Vector3, scale: Float, level: Float, material: Material, primitiveType: Int
  ): Geometry & FractionalLevelSponge

  private[objects] lazy val transparentSponge: Option[Geometry & FractionalLevelSponge] =
    if level.isValidInt then None
    else Some(createInstance(center, scale, level.floor, transparentMaterial, primitiveType))

  private[objects] lazy val nextLevelSponge: Option[Geometry & FractionalLevelSponge] =
    if level.isValidInt then None
    else Some(createInstance(center, scale, (level+1).floor, material, primitiveType))

object FractionalLevelSponge:
  /** Absolute world-space offset applied outward along normals to skin faces in fractional-level
   *  rendering, to prevent z-fighting with the underlying sponge faces at non-hole positions.
   *  The skin mesh is expanded by this amount so the continuation ray (tmin =
   *  COVERAGE_CONTINUATION_OFFSET = 0.0001f) can reach the sponge face just behind it.
   *  Value = 3 * COVERAGE_CONTINUATION_OFFSET = 0.0003f: sub-pixel at typical renders,
   *  ~3% of sub-cube width at level 4, ~9% at level 5. */
  val SkinNormalOffset: Float = 0.0003f
