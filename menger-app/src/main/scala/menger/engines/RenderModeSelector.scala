package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType

enum SceneType:
  case TriangleMeshes(specs: List[ObjectSpec])
  case SimpleMixed(specs: List[ObjectSpec], meshType: String)
  case Menger4D(specs: List[ObjectSpec])
  case Sierpinski4D(specs: List[ObjectSpec])
  case Hexadecachoron4D(specs: List[ObjectSpec])
  case Curves(specs: List[ObjectSpec])
  case Unsupported(specs: List[ObjectSpec])

/** Classifies a list of ObjectSpecs into a SceneType used by BaseEngine
  * to decide how to split and build the scene.
  *
  * Delegates to [[TypeRegistry]] for type → SceneType mapping (T1, Sprint 32).
  */
object RenderModeSelector:

  def classify(specs: List[ObjectSpec]): SceneType =
    require(specs.nonEmpty, "classify requires at least one ObjectSpec")
    val types = specs.map(s => ObjectType.normalize(s.objectType)).toSet

    // If all specs share the same built-in type, use TypeRegistry
    if types.size == 1 && TypeRegistry.builtInTypeNames.contains(types.head) then
      TypeRegistry.forType(types.head).get.sceneTypeFactory(specs)
    else
      val hasAnalytical = types.exists(ObjectType.isAnalyticalPrimitive)
      val hasCubeSponge = types.contains("cube-sponge")
      val otherTypes    = types.filterNot(t =>
        ObjectType.isAnalyticalPrimitive(t) || t == "cube-sponge")

      if types.forall(_ == "curve") then
        SceneType.Curves(specs)
      else if types.forall(ObjectType.isTriangleMesh) then
        SceneType.TriangleMeshes(specs)
      else if (hasAnalytical || hasCubeSponge) && otherTypes.forall(ObjectType.isTriangleMesh) then
        val tag = otherTypes.headOption.getOrElse(
          if hasCubeSponge then "cube-sponge"
          else types.find(ObjectType.isAnalyticalPrimitive).get
        )
        SceneType.SimpleMixed(specs, tag)
      else if types.forall(ObjectType.isMenger4D) then
        SceneType.Menger4D(specs)
      else if types.forall(ObjectType.isSierpinski4D) then
        SceneType.Sierpinski4D(specs)
      else if types.forall(ObjectType.isHexadecachoron4D) then
        SceneType.Hexadecachoron4D(specs)
      else
        SceneType.Unsupported(specs)
