package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType

/**
 * Classifies a list of ObjectSpecs into a SceneType used by BaseEngine
 * to decide how to split and build the scene.
 *
 * Builder selection is delegated to GeometryRegistry.
 */
object RenderModeSelector:

  def classify(specs: List[ObjectSpec]): SceneType =
    require(specs.nonEmpty, "classify requires at least one ObjectSpec")
    val types = specs.map(s => ObjectType.normalize(s.objectType)).toSet

    val hasAnalytical = types.exists(ObjectType.isAnalyticalPrimitive)
    val hasCubeSponge = types.contains("cube-sponge")
    val otherTypes    = types.filterNot(t =>
      ObjectType.isAnalyticalPrimitive(t) || t == "cube-sponge")

    if !hasAnalytical && !hasCubeSponge && otherTypes.forall(ObjectType.isTriangleMesh) then
      SceneType.TriangleMeshes(specs)
    else if (hasAnalytical || hasCubeSponge) && otherTypes.forall(ObjectType.isTriangleMesh) then
      val tag = otherTypes.headOption.getOrElse(
        if hasCubeSponge then "cube-sponge"
        else types.find(ObjectType.isAnalyticalPrimitive).get
      )
      SceneType.SimpleMixed(specs, tag)
    else
      SceneType.Unsupported(specs)

enum SceneType:
  case TriangleMeshes(specs: List[ObjectSpec])
  case SimpleMixed(specs: List[ObjectSpec], meshType: String)
  case Unsupported(specs: List[ObjectSpec])
