package menger.engines

import menger.ObjectSpec
import menger.ProfilingConfig
import menger.common.ObjectType
import menger.engines.scene.CubeSpongeSceneBuilder
import menger.engines.scene.SceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder

/** Scene classification and builder selection logic shared by OptiXEngine and AnimatedOptiXEngine. */
object SceneClassifier:

  def classify(specs: List[ObjectSpec]): SceneType =
    require(specs.nonEmpty, "classify requires at least one ObjectSpec")
    val types = specs.map(_.objectType.toLowerCase).toSet

    if types == Set("cube-sponge") then
      SceneType.CubeSponges(specs)
    else if types.forall(_ == "sphere") then
      SceneType.Spheres(specs)
    else if !types.contains("sphere") && !types.contains("cube-sponge") &&
            types.forall(isTriangleMeshType) then
      SceneType.TriangleMeshes(specs)
    else
      // Mixed scene — spheres and/or cube-sponges combined with triangle meshes.
      // Sprint 18.1 TD-5 resolution lets per-spec setTriangleMesh +
      // addTriangleMeshInstance create one GAS per mesh in the IAS; the engine
      // splits SimpleMixed into spheres / cube-sponges / other meshes and runs
      // the matching builder on each group (H-sponge-showcase-crash fix).
      val hasSpheres     = types.contains("sphere")
      val hasCubeSponge  = types.contains("cube-sponge")
      val otherMeshTypes = types - "sphere" - "cube-sponge"

      if (hasSpheres || hasCubeSponge) && otherMeshTypes.forall(isTriangleMeshType) then
        val tag = otherMeshTypes.headOption.getOrElse(
          if hasCubeSponge then "cube-sponge" else "sphere"
        )
        SceneType.SimpleMixed(specs, tag)
      else
        SceneType.ComplexMixed(specs)

  def isTriangleMeshType(objectType: String): Boolean =
    val t = objectType.toLowerCase
    t == "cube" ||
    t == "parametric" ||
    ObjectType.isSponge(t) ||
    ObjectType.isProjected4D(t)

  def selectSceneBuilder(
    sceneType: SceneType,
    textureDir: Option[String],
    gpuProject4D: Boolean = false
  )(using ProfilingConfig): Option[SceneBuilder] =
    val dir = textureDir.getOrElse(".")
    sceneType match
      case SceneType.Spheres(_)            => Some(SphereSceneBuilder())
      case SceneType.CubeSponges(_)        => Some(CubeSpongeSceneBuilder())
      case SceneType.TriangleMeshes(specs) => Some(selectTriangleMeshBuilder(specs, dir, gpuProject4D))
      case SceneType.SimpleMixed(_, _)     => None  // Handled specially in createMultiObjectScene
      case SceneType.ComplexMixed(_)       => None

  private def selectTriangleMeshBuilder(
    specs: List[ObjectSpec], dir: String, gpuProject4D: Boolean
  )(using ProfilingConfig): SceneBuilder =
    val all4DProjected   = specs.forall(s => ObjectType.isProjected4D(s.objectType))
    val hasEdgeRendering = specs.exists(_.hasEdgeRendering)
    if all4DProjected && hasEdgeRendering then TesseractEdgeSceneBuilder(dir)
    else TriangleMeshSceneBuilder(dir, gpuProject4D)

enum SceneType:
  case CubeSponges(specs: List[ObjectSpec])
  case Spheres(specs: List[ObjectSpec])
  case TriangleMeshes(specs: List[ObjectSpec])
  case SimpleMixed(specs: List[ObjectSpec], meshType: String)
  case ComplexMixed(specs: List[ObjectSpec])
