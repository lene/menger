package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType

/** How a mixed scene's specs are split into groups that one scene builder each can build.
  * Shared by the engines (`BaseEngine`) and `SceneValidator`, so the validator checks exactly
  * the grouping the renderer is going to build (usability review 2026-09, F19). */
object SceneGroups:

  /** Some, but not all, specs render edges. No single builder takes such a list:
    * `TesseractEdgeSceneBuilder` needs edges on every spec, and the other triangle-mesh
    * builders ignore them (usability review 2026-09, F18). */
  def hasMixedEdge4D(specs: List[ObjectSpec]): Boolean =
    specs.exists(_.hasEdgeRendering) && !specs.forall(_.hasEdgeRendering)

  /** The groups in build order. Edge-rendered objects come first because their builders may
    * reinitialize the renderer, which discards every instance built before it -- an
    * analytical sphere built earlier simply vanished (usability review 2026-09, F24): the 4D
    * ones, then the 3D meshes with edge tubes (menger#132; every group gets the same budget, so
    * only the first of them grows the renderer). Then one group per analytical primitive type,
    * one per 4D IFS fractal type (each has its own builder, menger#134), the cube sponges, and
    * all remaining meshes. */
  def buildOrder(specs: List[ObjectSpec]): List[List[ObjectSpec]] =
    val (analytical, meshes) = specs.partition(s => ObjectType.isAnalyticalPrimitive(s.objectType))
    val (edge4D, nonEdge4D) = meshes.partition(s =>
      s.hasEdgeRendering && ObjectType.isProjected4D(s.objectType))
    val (edge3D, nonEdge) = nonEdge4D.partition(menger.engines.scene.MeshEdges3D.appliesTo)
    val (ifs4D, nonIFS) = nonEdge.partition(s => ObjectType.isIFS4D(s.objectType))
    val (cubeSponges, otherMeshes) = nonIFS.partition(_.objectType.toLowerCase == "cube-sponge")
    def byType(group: List[ObjectSpec]) =
      group.groupBy(s => ObjectType.normalize(s.objectType)).toList.sortBy(_._1).map(_._2)
    (edge4D :: edge3D :: byType(analytical) ++ byType(ifs4D) ++ List(cubeSponges, otherMeshes))
      .filter(_.nonEmpty)
