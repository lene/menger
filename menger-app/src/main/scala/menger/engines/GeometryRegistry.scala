package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.engines.scene.ConeSceneBuilder
import menger.engines.scene.CubeSpongeSceneBuilder
import menger.engines.scene.CurveSceneBuilder
import menger.engines.scene.Hexadecachoron4DSceneBuilder
import menger.engines.scene.Menger4DSceneBuilder
import menger.engines.scene.PlaneSceneBuilder
import menger.engines.scene.SceneBuilder
import menger.engines.scene.Sierpinski4DSceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder

/**
 * Central registry mapping geometry type sets to scene builder factories.
 *
 * To add a new geometry type:
 *   1. Add the type string to ObjectType.VALID_TYPES
 *   2. Add isTriangleMesh coverage in ObjectType.isTriangleMesh
 *   3. No engine modification required — TriangleMeshSceneBuilder handles it
 *
 * For types that need a dedicated builder (like sphere or cube-sponge),
 * add an explicit branch in builderFor.
 *
 * Mixed groups (sphere+cube etc.) return None; callers must split first.
 */
object GeometryRegistry:

  /**
   * Returns the scene builder for the given homogeneous group of specs,
   * or None if the types are mixed/unknown and no single builder can handle them.
   */
  def builderFor(
    specs: List[ObjectSpec],
    textureDir: String = "."
  )(using ProfilingConfig): Option[SceneBuilder] =
    if specs.isEmpty then None
    else
      val types = specs.map(s => ObjectType.normalize(s.objectType)).toSet
      if types.forall(_ == "sphere") then
        Some(SphereSceneBuilder(textureDir))
      else if types.forall(_ == "curve") then
        Some(CurveSceneBuilder(textureDir))
      else if types.forall(_ == "cube-sponge") then
        Some(CubeSpongeSceneBuilder(textureDir))
      else if types.forall(ObjectType.isTriangleMesh) then
        val all4DProjected = types.forall(ObjectType.isProjected4D)
        val hasEdge        = specs.exists(_.hasEdgeRendering)
        if all4DProjected && hasEdge then
          Some(TesseractEdgeSceneBuilder(textureDir))
        else
          Some(TriangleMeshSceneBuilder(textureDir))
      else if types.forall(_ == "cone") then
        Some(ConeSceneBuilder(textureDir))
      else if types.forall(_ == "plane") then
        Some(PlaneSceneBuilder(textureDir))
      else if types.forall(ObjectType.isMenger4D) then
        Some(Menger4DSceneBuilder(textureDir))
      else if types.forall(ObjectType.isSierpinski4D) then
        Some(Sierpinski4DSceneBuilder(textureDir))
      else if types.forall(ObjectType.isHexadecachoron4D) then
        Some(Hexadecachoron4DSceneBuilder(textureDir))
      else
        None
