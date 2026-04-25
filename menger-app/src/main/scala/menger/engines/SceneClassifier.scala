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

    if types.contains("cube-sponge") then
      SceneType.CubeSponges(specs)
    else if types.forall(_ == "sphere") then
      SceneType.Spheres(specs)
    else if types.forall(isTriangleMeshType) then
      SceneType.TriangleMeshes(specs)
    else
      // Mixed scene - spheres + triangle meshes
      val hasSpheres  = types.contains("sphere")
      val meshTypes   = types.filter(isTriangleMeshType)

      if hasSpheres && meshTypes.nonEmpty then
        // TD-5 resolution (Sprint 18.1): per-spec setTriangleMesh +
        // addTriangleMeshInstance creates one GAS per mesh in the IAS, so spheres
        // can coexist with any combination of triangle-mesh types in a single scene.
        SceneType.SimpleMixed(specs, meshTypes.head)
      else
        // Catch-all: scene contains a non-sphere, non-triangle-mesh, non-cube-sponge
        // type that doesn't fit any builder.
        SceneType.ComplexMixed(specs)

  def isTriangleMeshType(objectType: String): Boolean =
    val t = objectType.toLowerCase
    t == "cube" ||
    t == "parametric" ||
    ObjectType.isSponge(t) ||
    ObjectType.isProjected4D(t)

  def selectSceneBuilder(
    sceneType: SceneType,
    textureDir: Option[String]
  )(using ProfilingConfig): Option[SceneBuilder] =
    val dir = textureDir.getOrElse(".")
    sceneType match
      case SceneType.Spheres(_)            => Some(SphereSceneBuilder())
      case SceneType.CubeSponges(_)        => Some(CubeSpongeSceneBuilder())
      case SceneType.TriangleMeshes(specs) => Some(selectTriangleMeshBuilder(specs, dir))
      case SceneType.SimpleMixed(_, _)     => None  // Handled specially in createMultiObjectScene
      case SceneType.ComplexMixed(_)       => None

  private def selectTriangleMeshBuilder(specs: List[ObjectSpec], dir: String)(using ProfilingConfig): SceneBuilder =
    val all4DProjected   = specs.forall(s => ObjectType.isProjected4D(s.objectType))
    val hasEdgeRendering = specs.exists(_.hasEdgeRendering)
    if all4DProjected && hasEdgeRendering then TesseractEdgeSceneBuilder(dir)
    else TriangleMeshSceneBuilder(dir)

enum SceneType:
  case CubeSponges(specs: List[ObjectSpec])
  case Spheres(specs: List[ObjectSpec])
  case TriangleMeshes(specs: List[ObjectSpec])
  case SimpleMixed(specs: List[ObjectSpec], meshType: String)
  case ComplexMixed(specs: List[ObjectSpec])
