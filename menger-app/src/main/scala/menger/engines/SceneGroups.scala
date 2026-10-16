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

  /** The groups in build order. Edge-rendered 4D objects come first because their builder may
    * reinitialize the renderer, which discards every instance built before it -- an
    * analytical sphere built earlier simply vanished (usability review 2026-09, F24). Then one
    * group per analytical primitive type, the cube sponges, and all remaining meshes. */
  def buildOrder(specs: List[ObjectSpec]): List[List[ObjectSpec]] =
    val (analytical, meshes) = specs.partition(s => ObjectType.isAnalyticalPrimitive(s.objectType))
    val (edge4D, nonEdge) = meshes.partition(s =>
      s.hasEdgeRendering && ObjectType.isProjected4D(s.objectType))
    val (cubeSponges, otherMeshes) = nonEdge.partition(_.objectType.toLowerCase == "cube-sponge")
    val analyticalGroups = analytical.groupBy(_.objectType.toLowerCase).values.toList
    (edge4D :: analyticalGroups ++ List(cubeSponges, otherMeshes)).filter(_.nonEmpty)
