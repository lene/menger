package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.engines.scene.SceneBuilder

/** Central registry mapping geometry type sets to scene builder factories.
  *
  * Delegates to [[TypeRegistry]] — the single source of truth for
  * type → builder mappings (T1, Sprint 32).
  *
  * Mixed groups (sphere+cube etc.) return None; callers must split first.
  */
object GeometryRegistry:

  /** Returns the scene builder for the given homogeneous group of specs,
    * or None if the types are mixed/unknown and no single builder can handle them.
    */
  def builderFor(
    specs: List[ObjectSpec],
    textureDir: String = "."
  )(using pc: ProfilingConfig): Option[SceneBuilder] =
    if specs.isEmpty then return None

    val types = specs.map(s => ObjectType.normalize(s.objectType)).toSet

    // Special case: 4D projected triangle meshes with edge rendering
    val all4DProjected = types.forall(ObjectType.isProjected4D)
    val hasEdge = specs.exists(_.hasEdgeRendering)
    if types.forall(ObjectType.isTriangleMesh) && all4DProjected && hasEdge then
      return Some(menger.engines.scene.TesseractEdgeSceneBuilder(textureDir)(using pc))

    // If all specs share the SAME type, delegate to TypeRegistry
    if types.size == 1 then
      val typeName = types.head
      TypeRegistry.forType(typeName).map { entry =>
        entry.builderFactory(textureDir, pc)
      }
    else
      // Mixed types — all must be triangle meshes
      if types.forall(ObjectType.isTriangleMesh) then
        Some(menger.engines.scene.TriangleMeshSceneBuilder(textureDir)(using pc))
      else
        None
